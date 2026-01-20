# main.py (デイトレード最適化版)
# Hyperliquid 自動トレーディングボット (Google Sheets統合版 - Gemini API使用)

import os
import sys
import time
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from hyperliquid_sdk_trader import HyperliquidSDKTrader
from advanced_market_data import AdvancedMarketData
from risk_manager import RiskManager
from google_sheets_logger import GoogleSheetsLogger
from ml_predictor import MLPredictor
from online_learning import OnlineLearner

load_dotenv()

# 緊急損切り・利確設定
EMERGENCY_SL_PCT = float(os.getenv('EMERGENCY_STOP_LOSS', '-2.0')) # デイトレ用にタイトに設定
SECURE_PROFIT_TP_PCT = float(os.getenv('SECURE_TAKE_PROFIT', '4.0'))
MIN_SIGNAL_STRENGTH = int(os.getenv('MIN_SIGNAL_STRENGTH', '60'))

# 時間軸設定
MAIN_TIMEFRAME = '15m'  # デイトレードの主軸
TREND_TIMEFRAME = '1h'  # 環境認識用

class TradingBot:
    """
    Hyperliquid 自動トレーディングボット (デイトレード特化版)
    LightGBM + LSTM によるアンサンブル予測 + 板情報分析
    """
    
    def __init__(self, symbol='ETH', initial_capital=1000.0, enable_sheets_logging=True):
        self.network      = os.getenv("NETWORK", "testnet").lower()
        self.bot_name     = "Mainnet" if self.network == "mainnet" else "Testnet"
        self.symbol       = symbol
        self.trader       = HyperliquidSDKTrader()
        self.market_data  = AdvancedMarketData(f'{symbol}-USD')
        self.risk_manager = RiskManager(initial_capital)
        self.running      = False
        self.enable_sheets_logging = enable_sheets_logging
        
        # エントリー時刻管理（時間切れ撤退用）
        self.last_entry_time = None

        # 機械学習予測器
        self.ml_predictor = MLPredictor(symbol=symbol)
        # 15分足ベースで学習するように設定
        self.online_learner = OnlineLearner(symbol=symbol, timeframe=MAIN_TIMEFRAME, retrain_interval_hours=24)
        print(f"🤖 機械学習予測システム: 有効 (Timeframe: {MAIN_TIMEFRAME})")
        print(f"   モデル状態: {self.ml_predictor.lgb_model is not None or self.ml_predictor.lstm_model is not None}")
        
        # Google Sheetsロガー初期化
        self.sheets_logger = None
        if self.enable_sheets_logging:
            try:
                self.sheets_logger = GoogleSheetsLogger()
                print(f"📊 Google Sheetsログ記録: 有効")
            except Exception as e:
                print(f"⚠️ Google Sheetsログ記録を無効化: {e}")
                self.enable_sheets_logging = False
        
        print("\n" + "="*70)
        print(f"🚀 Hyperliquid {self.bot_name} Bot (DayTrade Logic)")
        print("="*70)

    
    def get_ml_decision(self, market_analysis: dict, account_state: dict, structure_data: dict) -> dict:
        """
        【修正・最適化版】機械学習ベースの取引判断
        - 15分足データを使用
        - 板情報の不均衡(Imbalance)を考慮
        - 時間経過による撤退ロジックを追加
        """
        try:
            # === ステップ1: データ取得 (15分足) ===
            df_main = self.market_data.get_ohlcv(MAIN_TIMEFRAME, limit=200)
            
            # 板情報の偏りを取得
            imbalance = structure_data.get('orderbook_imbalance', 0)
            
            # === ステップ2: ML予測実行 (板情報を注入) ===
            ml_result = self.ml_predictor.predict(df_main, extra_features={'imbalance': imbalance})
            
            # モデル未学習時のガード
            if ml_result.get('model_used') == 'NONE':
                return {
                    'action': 'HOLD', 'side': 'NONE', 'confidence': 0,
                    'reasoning': ml_result.get('reasoning', 'モデル未学習')
                }
            
            # === ステップ3: 確率分布の解析 ===
            up_prob = ml_result['up_prob']
            down_prob = ml_result['down_prob']
            confidence = ml_result['confidence']
            
            # 既存ポジションの特定
            existing_side = None
            if account_state and 'assetPositions' in account_state:
                for pos in account_state['assetPositions']:
                    p = pos.get('position', {})
                    if p.get('coin') == self.symbol and float(p.get('szi', 0)) != 0:
                        existing_side = 'LONG' if float(p.get('szi', 0)) > 0 else 'SHORT'
                        break
            
            # --- 閾値設定 (デイトレ用) ---
            ENTRY_THRESHOLD = 0.45  # 少し厳しめに
            CLOSE_THRESHOLD = 0.40  # 逆行したら早めに逃げる

            action = 'HOLD'
            side = 'NONE'
            reasoning = f"Wait: Up({up_prob:.2f}) Down({down_prob:.2f})"

            # --- 板情報フィルター ---
            # 買いシグナルだが、板が売り圧(マイナス)なら見送る
            if imbalance < -0.3 and up_prob > down_prob:
                reasoning += " (板情報により買い見送り)"
                confidence = 0 # 自信度を下げる
            elif imbalance > 0.3 and down_prob > up_prob:
                reasoning += " (板情報により売り見送り)"
                confidence = 0

            if existing_side:
                # === 決済ロジック (逆行シグナルで撤退) ===
                if existing_side == 'LONG' and down_prob > CLOSE_THRESHOLD:
                    action = 'CLOSE'
                    reasoning = f'LONG決済: 下落予測優勢 ({down_prob*100:.1f}%)'
                elif existing_side == 'SHORT' and up_prob > CLOSE_THRESHOLD:
                    action = 'CLOSE'
                    reasoning = f'SHORT決済: 上昇予測優勢 ({up_prob*100:.1f}%)'
                
                # === 時間切れ撤退 (Time-based Exit) ===
                # エントリーから3時間経過しても決済条件にかからない場合は手仕舞い
                if self.last_entry_time and (datetime.now() - self.last_entry_time).total_seconds() > 3 * 3600:
                    action = 'CLOSE'
                    reasoning = 'TimeExit: ポジション滞留時間超過 (3時間)'

            else:
                # === 新規エントリーロジック ===
                if up_prob >= ENTRY_THRESHOLD and up_prob > down_prob:
                    action = 'BUY'
                    side = 'LONG'
                    reasoning = f'上昇予測: {up_prob*100:.1f}% (Board: {imbalance:.2f})'
                elif down_prob >= ENTRY_THRESHOLD and down_prob > up_prob:
                    action = 'SELL'
                    side = 'SHORT'
                    reasoning = f'下落予測: {down_prob*100:.1f}% (Board: {imbalance:.2f})'
            
            # === ステップ4: 動的リスクパラメータ (ボラティリティ連動) ===
            volatility = market_analysis.get('volatility', 2.0)
            
            # ボラティリティに応じたSL/TP設定 (デイトレード用・やや浅め)
            if volatility > 3.0: # 高ボラ
                sl_pct, tp_pct = 2.0, 4.0
            else: # 通常
                sl_pct, tp_pct = 1.0, 1.5
            
            # 期待値 (EV) の概算
            win_prob = up_prob if action == 'BUY' else down_prob if action == 'SELL' else 0.0
            if action in ['BUY', 'SELL']:
                expected_value_r = (win_prob * tp_pct) - ((1 - win_prob) * sl_pct)
            else:
                expected_value_r = 0

            # === 最終結果 ===
            print(f"\n🤖 ML判断詳細:")
            print(f"   Model: {ml_result['model_used']}")
            print(f"   Prob: Up {up_prob*100:.1f}% | Down {down_prob*100:.1f}%")
            print(f"   Board Imbalance: {imbalance:.2f}")
            print(f"   Action: {action} (Conf: {confidence})")

            return {
                'action': action,
                'side': side,
                'confidence': confidence,
                'expected_value_r': expected_value_r,
                'risk_reward_ratio': tp_pct / sl_pct,
                'stop_loss_percent': sl_pct,
                'take_profit_percent': tp_pct,
                'reasoning': f"{reasoning} | {ml_result['model_used']}",
                'ml_probabilities': {'up': up_prob, 'down': down_prob}
            }
            
        except Exception as e:
            print(f"⚠️ ML判断エラー: {e}")
            import traceback
            traceback.print_exc()
            return {'action': 'HOLD', 'side': 'NONE', 'confidence': 0, 'reasoning': f'Error: {str(e)}'}
    

    
    def log_to_sheets(self, trade_data: dict = None, signal_data: dict = None, snapshot_data: dict = None):
        """
        Google Sheetsにログを記録
        """
        if not self.enable_sheets_logging or not self.sheets_logger:
            return
        
        try:
            # 1. 実行履歴 (Executions)
            if trade_data:
                self.sheets_logger.log_execution(trade_data)
            
            # 2. AI分析 (AI_Analysis)
            if signal_data:
                probs = signal_data.get('ml_probabilities', {})
                analysis_payload = {
                    'timestamp': signal_data.get('timestamp'),
                    'price': signal_data.get('price'),
                    'action': signal_data.get('action', 'HOLD'),
                    'confidence': signal_data.get('confidence', 0),
                    'up_prob': probs.get('up', 0),
                    'down_prob': probs.get('down', 0),
                    'market_regime': signal_data.get('market_regime', 'NORMAL'),
                    'model_used': signal_data.get('model_used', 'ENSEMBLE'),
                    'rsi': signal_data.get('rsi', 0),
                    'volatility': signal_data.get('volatility', 0)
                }
                self.sheets_logger.log_ai_analysis(analysis_payload)
            
            # 3. 資産推移 (Equity)
            if snapshot_data:
                pos_val = snapshot_data.get('position_size', 0) * snapshot_data.get('eth_price', 0)
                equity_payload = {
                    'timestamp': snapshot_data.get('timestamp'),
                    'account_value': snapshot_data.get('account_value'),
                    'available_balance': snapshot_data.get('available_balance'),
                    'position_value': pos_val,
                    'unrealized_pnl': snapshot_data.get('unrealized_pnl', 0),
                    'realized_pnl_cumulative': snapshot_data.get('realized_pnl_cumulative', 0)
                }
                self.sheets_logger.log_equity(equity_payload)
                
        except Exception as e:
            print(f"⚠️ Google Sheetsログ記録エラー: {e}")
            
    
    def execute_trade(self, decision: dict, current_price: float, account_state: dict, analysis: dict):
        """
        実際の取引を実行してGoogle Sheetsに記録
        """
        action = decision.get('action')

        # === 1. EV/RRチェック (BUY/SELLのみ) ===
        ev = float(decision.get('expected_value_r', 0))
        rr_ratio = float(decision.get('risk_reward_ratio', 0))
        
        if action in ['BUY', 'SELL']:
            if ev <= 0.4: # デイトレ用に少し緩和
                print(f"🛑 取引拒否: 期待値不足 (EV: {ev:.2f})")
                return
            if rr_ratio < 1.2: # デイトレ用に少し緩和
                print(f"🛑 取引拒否: リスクリワード比不足 (RR: {rr_ratio:.2f})")
                return
        
        # === 2. アカウント情報・既存ポジション一括取得 ===
        cross_margin = account_state.get('crossMarginSummary', {}) if account_state else {}
        margin_summary = account_state.get('marginSummary', {}) if account_state else {}
        account_value = float(cross_margin.get('accountValue', 0)) or float(margin_summary.get('accountValue', 0))
        available_balance = float(cross_margin.get('totalRawUsd', 0)) or float(margin_summary.get('totalRawUsd', 0))
        
        self.risk_manager.current_capital = account_value
        pos_data = self._get_position_summary(account_state)
        existing_position_value = pos_data['position_value']
        unrealized_pnl = pos_data['unrealized_pnl']
        
        # === 3. 日次損失制限チェック ===
        if not self.risk_manager.check_daily_loss_limit():
            print("🛑 日次損失限度に達したため取引を見送ります")
            return
        
        # === 4. AI自信度を取得 ===
        confidence = float(decision.get('confidence', 0))
        
        # === 5. 追加ポジション可否判定 (CLOSE以外) ===
        if action != 'CLOSE' and existing_position_value > 0:
            if not self.risk_manager.should_add_position(confidence, existing_position_value):
                print(f"⚠️ 既存ポジション${existing_position_value:.2f}あり、自信度{confidence:.0f}%では追加不可")
                return
        
        # === 6. SL/TP/Side取得 ===
        sl_percent = float(decision.get('stop_loss_percent', 2.0))
        tp_percent = float(decision.get('take_profit_percent', 3.0))
        side = decision.get('side')
        
        # === 7. ポジションサイズ計算 ===
        if action == 'CLOSE':
            size = 0.0
            risk_level = "CLOSE"
            reasoning = decision.get('reasoning')
            order_value = 0.0
        else:
            print(f"\n{'='*70}")
            print(f"🔍 AI自信度ベースのポジションサイズ計算")
            print(f"{'='*70}")
            
            position_result = self.risk_manager.calculate_position_size_by_confidence(
                capital=account_value,    
                entry_price=current_price,
                confidence=confidence,
                existing_position_value=existing_position_value,
                stop_loss_percent=sl_percent,
                max_available_cash=available_balance
            )
            
            size = position_result['size']
            risk_level = position_result['risk_level']
            reasoning = position_result['reasoning']
            order_value = position_result['position_value']
            
            print(f"\n✅ 計算結果:")
            print(f"   ポジションサイズ: {size:.4f} ETH")
            print(f"   ポジション金額: ${order_value:.2f}")
            print(f"   リスクレベル: {risk_level}")
            print(f"{'='*70}\n")
            
            if size == 0:
                print(f"⚠️ ポジションサイズがゼロのため取引見送り")
                return

        # === 8. 注文実行 ===
        trade_success = False
        estimated_fee = 0.0

        if action == 'CLOSE':
            print(f"📉 ポジション決済実行...")
            result = self.trader.close_position(self.symbol)
            trade_success = result and result.get('status') == 'ok'
            if trade_success:
                self.last_entry_time = None # エントリー時刻リセット
        else:
            # エントリー時刻を記録
            self.last_entry_time = datetime.now()
            
            # SL/TP価格計算 (ログ用)
            stop_loss_price = self.risk_manager.calculate_stop_loss(current_price, side, percent=sl_percent)
            take_profit_price = self.risk_manager.calculate_take_profit(current_price, stop_loss_price, rr_ratio)
            
            risk_summary = self.risk_manager.get_risk_summary(current_price, size, stop_loss_price, take_profit_price, 1)
            print(f"📊 リスク: ${risk_summary['risk_amount']:.2f} / リワード: ${risk_summary['reward_amount']:.2f}")

            print(f"🛡️ 指値注文を送信中...")
            is_buy = (side == 'LONG')
            result = self.trader.place_limit_order(
                symbol=self.symbol,
                is_buy=is_buy,
                size=size,
                time_in_force="Ioc", 
                aggressive=True 
            )
            estimated_fee = order_value * 0.00035
            trade_success = result and result.get('status') == 'ok'

        if trade_success:
            print("✅ 取引成功!")
            if action != 'CLOSE':
                self.risk_manager.update_position_tracking(order_value, "ADD")
            else:
                self.risk_manager.update_position_tracking(0, "CLOSE")
        else:
            print("❌ 取引失敗")
        
        # === 9. Google Sheetsログ記録 ===
        self.log_to_sheets(
            trade_data={
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': action,
                'side': side,
                'size': size,
                'price': current_price,
                'order_value': order_value,
                'fee': estimated_fee if trade_success else 0,
                'realized_pnl': 0,
                'unrealized_pnl': unrealized_pnl, 
                'confidence': confidence,
                'signal_strength': analysis.get('signal_strength', 0),
                'leverage': 1,
                'balance': available_balance,
                'reasoning': reasoning,
                'status': 'EXECUTED' if trade_success else 'FAILED'
            },
            signal_data={
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': action,
                'confidence': confidence,
                'ml_probabilities': decision.get('ml_probabilities', {}),
                'price': current_price,
                'volatility': analysis.get('volatility', 0),
                'rsi': analysis.get('indicators', {}).get('rsi', 0),
                'market_regime': decision.get('market_regime', 'NORMAL'),
                'model_used': decision.get('reasoning', '').split('|')[-1].strip()
            },
            snapshot_data={
                'timestamp': datetime.now(),
                'account_value': account_value,
                'available_balance': available_balance,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl_cumulative': 0,
                'eth_price': current_price,
                'position_size': size if trade_success and action != 'CLOSE' else 0,
                'action': action,
                'confidence': confidence,
                'total_trades': 0,
                'notes': f"{action} {side} | {risk_level}"
            }
        )
    
    def check_daily_exit(self, account_state: dict):
        """
        日次強制リセット (日本時間 朝8:55 = UTC 23:55)
        Funding Rate支払いや日またぎリスクを回避
        """
        now = datetime.utcnow()
        # UTC 23:55 (JST 08:55)
        if now.hour == 23 and now.minute >= 55:
            pos_data = self._get_position_summary(account_state)
            if pos_data['found']:
                print("\n" + "!"*70)
                print("⏰ 日次強制決済時刻 (UTC 23:55)")
                print("   全ポジションをクローズして日またぎリスクを回避します")
                print("!"*70 + "\n")
                
                self.trader.close_position(self.symbol)
                self.last_entry_time = None
                
                # ログ記録
                self.log_to_sheets(trade_data={
                    'timestamp': datetime.now(),
                    'symbol': self.symbol,
                    'action': 'CLOSE',
                    'side': 'NONE',
                    'size': 0,
                    'price': 0,
                    'order_value': 0,
                    'fee': 0,
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'confidence': 0,
                    'signal_strength': 0,
                    'leverage': 0,
                    'balance': 0,
                    'reasoning': 'Daily Force Close',
                    'status': 'EXECUTED'
                })
                
                # 日が変わるまで待機
                print("⏳ 翌日まで待機中...")
                time.sleep(300) 

    def run_trading_loop(self, interval=60):
        """
        自動取引ループ
        """
        self.running = True
        self.online_learner.start_background_learning()
        
        print(f"\n🚀 自動トレーディング開始")
        print(f"   判断間隔: {interval}秒")
        print(f"   メイン時間軸: {MAIN_TIMEFRAME}")
        
        try:
            last_ai_check_time = 0
            fast_interval = 10 
            
            while self.running:
                current_time = time.time()

                # --- 高速監視フェーズ (10秒ごと: 価格と緊急停止) ---
                current_price = self.trader.get_current_price(self.symbol)
                account_state = self.trader.get_user_state()
                
                if not current_price:
                    time.sleep(fast_interval)
                    continue

                if account_state:
                    # 資産情報の更新
                    cross_margin = account_state.get('crossMarginSummary', {})
                    margin_summary = account_state.get('marginSummary', {})
                    account_value = float(cross_margin.get('accountValue', 0)) or float(margin_summary.get('accountValue', 0))
                    self.risk_manager.current_capital = account_value

                    # 日次リセットチェック
                    self.check_daily_exit(account_state)

                    # 緊急決済チェック
                    pos_data = self._get_position_summary(account_state)
                    if pos_data['found']:
                        self._check_emergency_exit(pos_data, current_price)

                # --- AI判断フェーズ (interval秒ごと) ---
                if (current_time - last_ai_check_time >= interval) or (last_ai_check_time == 0):
                    
                    # モデルのホットリロード
                    if self.ml_predictor:
                         try: self.ml_predictor.load_models()
                         except: pass

                    print(f"\n{'='*70}")
                    print(f"📊 {self.symbol} Price: ${current_price:.2f}")
                    
                    # 1. 市場分析データの取得
                    analysis = self.market_data.get_comprehensive_analysis()
                    
                    # 2. 板情報 (Structure) の取得
                    structure = self.market_data.get_market_structure_features()
                    imbalance = structure.get('orderbook_imbalance', 0)
                    
                    if analysis:
                        volatility = analysis.get('volatility', 0)
                        print(f"   Vol: {volatility:.2f}% | Board Imbalance: {imbalance:.2f}")
                        
                        # 3. ML判断を実行 (板情報を渡す)
                        decision = self.get_ml_decision(analysis, account_state, structure)
                        
                        if decision:
                            action = decision.get('action', 'HOLD')
                            confidence = decision.get('confidence', 0)
                            
                            # 4. AI思考ログの作成
                            signal_log = {
                                'timestamp': datetime.now(),
                                'symbol': self.symbol,
                                'action': action,
                                'confidence': confidence,
                                'ml_probabilities': decision.get('ml_probabilities'),
                                'price': current_price,
                                'volatility': volatility,
                                'rsi': analysis.get('indicators', {}).get('rsi', 0),
                                'market_regime': decision.get('market_regime'),
                                'model_used': decision.get('reasoning', '').split('|')[-1].strip()
                            }

                            # 5. 取引実行 または ログ記録のみ
                            if action == "CLOSE":
                                self.execute_trade(decision, current_price, account_state, analysis)
                            
                            elif action in ['BUY', 'SELL']:
                                # 閾値を 35% に緩和 (デイトレ用)
                                if confidence >= 35:
                                    self.execute_trade(decision, current_price, account_state, analysis)
                                else:
                                    print(f"⏸️ 信頼度不足で見送り ({confidence}%)")
                                    self.log_to_sheets(signal_data=signal_log)
                            else:
                                self.log_to_sheets(signal_data=signal_log)

                        else:
                            print("⚠️ ML判断不能")

                    last_ai_check_time = current_time
                    print(f"⏳ 待機中...")

                time.sleep(fast_interval)
                
        except KeyboardInterrupt:
            print("\n⏸️ 停止")
            self.online_learner.stop_background_learning()
            self.running = False

    def _check_emergency_exit(self, pos_data, current_price):
        """
        緊急決済ロジック
        """
        entry_px = pos_data['entry_price']
        side = pos_data['side']
        
        if side == 'LONG':
            pnl_pct = ((current_price - entry_px) / entry_px * 100)
        else:
            pnl_pct = ((entry_px - current_price) / entry_px * 100)
        
        if pnl_pct <= EMERGENCY_SL_PCT:
            print(f"🚨 緊急損切り: {pnl_pct:.2f}%")
            self.trader.close_position(self.symbol)
            self.risk_manager.update_position_tracking(0, "CLOSE")
            self.last_entry_time = None
        elif pnl_pct >= SECURE_PROFIT_TP_PCT:
            print(f"🎉 緊急利確: {pnl_pct:.2f}%")
            self.trader.close_position(self.symbol)
            self.risk_manager.update_position_tracking(0, "CLOSE")
            self.last_entry_time = None

    def _get_position_summary(self, account_state: dict) -> dict:
        """
        ポジション情報を一括取得 (entry_priceを追加)
        """
        summary = {
            'size': 0.0,
            'side': 'NONE',
            'unrealized_pnl': 0.0,
            'entry_price': 0.0,
            'position_value': 0.0,
            'found': False
        }

        if not account_state or 'assetPositions' not in account_state:
            return summary

        for pos in account_state['assetPositions']:
            item = pos.get('position', {})
            if item.get('coin') == self.symbol:
                szi = float(item.get('szi', 0))
                if szi == 0: continue

                size = abs(szi)
                entry_px = float(item.get('entryPx', 0))
                
                return {
                    'size': size,
                    'side': 'LONG' if szi > 0 else 'SHORT',
                    'unrealized_pnl': float(item.get('unrealizedPnl', 0)),
                    'entry_price': entry_px,
                    'position_value': size * entry_px,
                    'found': True
                }

        return summary


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    
    network = os.getenv("NETWORK", "testnet").lower()
    net_display = "MAINNET" if network == "mainnet" else "TESTNET"
    symbol = os.getenv('TRADING_SYMBOL', 'ETH')
    env_capital = os.getenv('INITIAL_CAPITAL', '1000')
    interval = int(os.getenv('CHECK_INTERVAL', '60'))
    enable_sheets = os.getenv('ENABLE_SHEETS_LOGGING', 'true').lower() == 'true'

    try:
        capital = float(env_capital)
    except ValueError:
        capital = 1000.0
    
    if mode == 'run':
        print(f"\n🚀 {net_display} モードで起動準備中...")
        try:
            temp_trader = HyperliquidSDKTrader()
            account_state = temp_trader.get_user_state()
            real_balance = 0.0
            if account_state:
                cross_margin = account_state.get('crossMarginSummary', {})
                margin_summary = account_state.get('marginSummary', {})
                real_balance = float(cross_margin.get('totalRawUsd', 0)) or float(margin_summary.get('totalRawUsd', 0))
            
            print(f"💳 ウォレット実残高 (Perps): ${real_balance:.2f}")
            print(f"⚙️ 設定された初期資金: ${capital:.2f}")
            
        except Exception as e:
            print(f"⚠️ 残高チェック時にエラー: {e}")
        
        bot = TradingBot(
            symbol=symbol, 
            initial_capital=capital,
            enable_sheets_logging=enable_sheets
        )
        bot.run_trading_loop(interval=interval)
    
    # 他のモード (test, buy, sell等) は省略せず残す場合はここに記述
    # 基本的には `python main.py` で動くようにしてあります

if __name__ == "__main__":
    main()