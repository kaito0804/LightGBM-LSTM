# main.py (修正版)
# Hyperliquid 自動トレーディングボット (Google Sheets統合版 - Gemini API使用)

import os
import sys
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from hyperliquid_sdk_trader import HyperliquidSDKTrader
from advanced_market_data import AdvancedMarketData
from risk_manager import RiskManager
from google_sheets_logger import GoogleSheetsLogger
from ml_predictor import MLPredictor
from online_learning import OnlineLearner

load_dotenv()

# 緊急損切り・利確設定
EMERGENCY_SL_PCT = float(os.getenv('EMERGENCY_STOP_LOSS', '-3.0'))
SECURE_PROFIT_TP_PCT = float(os.getenv('SECURE_TAKE_PROFIT', '6.0'))
MIN_SIGNAL_STRENGTH = int(os.getenv('MIN_SIGNAL_STRENGTH', '60'))

class TradingBot:
    """
    Hyperliquid 自動トレーディングボット (ML版)
    LightGBM + LSTM によるアンサンブル予測
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

        # 機械学習予測器
        self.ml_predictor = MLPredictor(symbol=symbol)
        self.online_learner = OnlineLearner(symbol=symbol, retrain_interval_hours=24)
        print(f"🤖 機械学習予測システム: 有効")
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
        print(f"🚀 Hyperliquid {self.bot_name} Bot (LightGBM/LSTM)")
        print("="*70)

    
    def get_ml_decision(self, market_analysis: dict, account_state: dict) -> dict:
        """
        【修正・最適化版】機械学習ベースの取引判断
        - 閾値を3値分類の実情に合わせて最適化 (0.5 -> 0.4)
        - データ取得期間を延長して計算精度向上
        """
        try:
            # === ステップ1: データ取得 ===
            # テクニカル指標の計算精度確保のため200本確保
            df_1h = self.market_data.get_ohlcv('1h', limit=200)
            
            # === ステップ2: ML予測実行 ===
            ml_result = self.ml_predictor.predict(df_1h)
            
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
            
            # --- 閾値設定 (3値分類用) ---
            # 中立があるため、0.40(40%)を超えれば方向性は明確と判断する
            ENTRY_THRESHOLD = 0.40  
            CLOSE_THRESHOLD = 0.45  # 反対方向がこれを越えたら逃げる

            action = 'HOLD'
            side = 'NONE'
            reasoning = f"Wait: Up({up_prob:.2f}) Down({down_prob:.2f})"

            if existing_side:
                # === 決済ロジック (逆行シグナルで撤退) ===
                if existing_side == 'LONG' and down_prob > CLOSE_THRESHOLD:
                    action = 'CLOSE'
                    reasoning = f'LONG決済: 下落予測優勢 ({down_prob*100:.1f}%)'
                elif existing_side == 'SHORT' and up_prob > CLOSE_THRESHOLD:
                    action = 'CLOSE'
                    reasoning = f'SHORT決済: 上昇予測優勢 ({up_prob*100:.1f}%)'
            else:
                # === 新規エントリーロジック ===
                # 確率が閾値を超え、かつ反対方向より大きい場合
                if up_prob >= ENTRY_THRESHOLD and up_prob > down_prob:
                    action = 'BUY'
                    side = 'LONG'
                    reasoning = f'上昇予測: {up_prob*100:.1f}%'
                elif down_prob >= ENTRY_THRESHOLD and down_prob > up_prob:
                    action = 'SELL'
                    side = 'SHORT'
                    reasoning = f'下落予測: {down_prob*100:.1f}%'
            
            # === ステップ4: 動的リスクパラメータ (ボラティリティ連動) ===
            volatility = market_analysis.get('volatility', 2.0)
            
            # ボラティリティに応じたSL/TP設定 (デイトレード用)
            if volatility > 5.0:   # 激しい相場
                sl_pct, tp_pct = 3.0, 5.0
            elif volatility > 3.0: # やや荒れ
                sl_pct, tp_pct = 2.0, 3.5
            else:                  # 通常・凪
                sl_pct, tp_pct = 1.5, 2.5
            
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
        【クリーンアップ版】Google Sheetsにログを記録
        - データ抽出ロジックを簡素化
        - Volatilityや確率データの取得漏れを防止
        """
        if not self.enable_sheets_logging or not self.sheets_logger:
            return
        
        try:
            # 1. 実行履歴 (Executions)
            if trade_data:
                self.sheets_logger.log_execution(trade_data)
            
            # 2. AI分析 (AI_Analysis)
            if signal_data:
                # 確率データの安全な抽出
                probs = signal_data.get('ml_probabilities', {})
                
                analysis_payload = {
                    'timestamp': signal_data.get('timestamp'),
                    'price': signal_data.get('price'),
                    # main.pyの決定アクションを優先
                    'action': signal_data.get('action', signal_data.get('recommendation', 'HOLD')),
                    'confidence': signal_data.get('confidence', 0),
                    'up_prob': probs.get('up', 0),
                    'down_prob': probs.get('down', 0),
                    'market_regime': signal_data.get('market_regime', 'NORMAL'),
                    'model_used': signal_data.get('model_used', 'ENSEMBLE'),
                    'rsi': signal_data.get('rsi', 0),
                    # ハードコード0を廃止し、渡された値を使用
                    'volatility': signal_data.get('volatility', 0)
                }
                self.sheets_logger.log_ai_analysis(analysis_payload)
            
            # 3. 資産推移 (Equity)
            if snapshot_data:
                # ポジション価値の計算（サイズ * 価格）
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
        ✅ 修正版: _get_position_summaryを活用してコードを大幅短縮
        """
        action = decision.get('action')

        # === 1. EV/RRチェック (BUY/SELLのみ) ===
        ev = float(decision.get('expected_value_r', 0))
        rr_ratio = float(decision.get('risk_reward_ratio', 0))
        
        if action in ['BUY', 'SELL']:
            if ev <= 0.5:
                print(f"🛑 取引拒否: 期待値不足 (EV: {ev:.2f} ≤ 0.5)")
                return
            if rr_ratio < 1.5:
                print(f"🛑 取引拒否: リスクリワード比不足 (RR: {rr_ratio:.2f} < 1.5)")
                return
        
        # === 2. アカウント情報・既存ポジション一括取得 ===
        # 資産情報の取得
        cross_margin = account_state.get('crossMarginSummary', {}) if account_state else {}
        margin_summary = account_state.get('marginSummary', {}) if account_state else {}
        account_value = float(cross_margin.get('accountValue', 0)) or float(margin_summary.get('accountValue', 0))
        available_balance = float(cross_margin.get('totalRawUsd', 0)) or float(margin_summary.get('totalRawUsd', 0))
        
        # Risk Manager更新
        self.risk_manager.current_capital = account_value

        # ✅ 【クリーンアップ】 ヘルパーメソッドで一発取得
        pos_data = self._get_position_summary(account_state)
        existing_position_value = pos_data['position_value']
        unrealized_pnl = pos_data['unrealized_pnl'] # ログ用に確保
        
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
        sl_percent = float(decision.get('stop_loss_percent', 3.0))
        tp_percent = float(decision.get('take_profit_percent', 5.0))
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
        else:
            # SL/TP価格計算 (ログ用)
            stop_loss_price = self.risk_manager.calculate_stop_loss(current_price, side, percent=sl_percent)
            take_profit_price = self.risk_manager.calculate_take_profit(current_price, stop_loss_price, rr_ratio)
            
            # リスクサマリー表示
            risk_summary = self.risk_manager.get_risk_summary(current_price, size, stop_loss_price, take_profit_price, 1)
            print(f"📊 リスク: ${risk_summary['risk_amount']:.2f} / リワード: ${risk_summary['reward_amount']:.2f}")

            # 指値注文 (IOC / Aggressive)
            print(f"🛡️ 指値注文を送信中...")
            is_buy = (side == 'LONG')
            result = self.trader.place_limit_order(
                symbol=self.symbol,
                is_buy=is_buy,
                size=size,
                time_in_force="Ioc", # 即時約定orキャンセル
                aggressive=True 
            )
            estimated_fee = order_value * 0.00035
            trade_success = result and result.get('status') == 'ok'

        if trade_success:
            print("✅ 取引成功!")
            if action != 'CLOSE':
                self.total_trades += 1
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
                'unrealized_pnl': unrealized_pnl, # ✅ ここもスッキリ
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
                'realized_pnl_cumulative': self.realized_pnl_cumulative,
                'eth_price': current_price,
                'position_size': size if trade_success and action != 'CLOSE' else 0,
                'action': action,
                'confidence': confidence,
                'total_trades': self.total_trades,
                'notes': f"{action} {side} | {risk_level}"
            }
        )
    
    def run_trading_loop(self, interval=60):
        """
        【修正・改善版】自動取引ループ
        - _get_position_summaryを活用してコードを短縮
        - AIの思考プロセス（HOLD含む）を全てGoogle Sheetsに記録
        """
        self.running = True
        self.online_learner.start_background_learning()
        
        print(f"\n🚀 自動トレーディング開始")
        print(f"   判断間隔: {interval}秒")
        
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

                    # ✅ 【修正】緊急決済チェック (ヘルパーメソッドで一発取得)
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
                    
                    if analysis:
                        signal_strength = analysis.get('signal_strength', 0)
                        volatility = analysis.get('volatility', 0)
                        
                        print(f"   テクニカルスコア: {signal_strength}/100 (Vol: {volatility:.2f}%)")
                        
                        # 2. ML判断を実行
                        decision = self.get_ml_decision(analysis, account_state)
                        
                        if decision:
                            action = decision.get('action', 'HOLD')
                            confidence = decision.get('confidence', 0)
                            
                            print(f"🎯 ML最終判断: {action} (信頼度: {confidence}%)")
                            
                            # 3. AI思考ログの作成
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

                            # 4. 取引実行 または ログ記録のみ
                            if action == "CLOSE":
                                self.execute_trade(decision, current_price, account_state, analysis)
                            
                            elif action in ['BUY', 'SELL']:
                                # 閾値を 40% に緩和 (get_ml_decisionですでにフィルタ済みのため)
                                if confidence >= 40:
                                    self.execute_trade(decision, current_price, account_state, analysis)
                                else:
                                    print(f"⏸️ 信頼度不足で見送り ({confidence}% < 40%)")
                                    # トレードしない場合も思考ログを残す
                                    self.log_to_sheets(signal_data=signal_log)
                            else:
                                # HOLDの場合もログを残す
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
        pos_data: _get_position_summary の戻り値 (整形済み) を受け取る
        """
        entry_px = pos_data['entry_price']
        side = pos_data['side']
        
        # PnL%計算
        if side == 'LONG':
            pnl_pct = ((current_price - entry_px) / entry_px * 100)
        else:
            pnl_pct = ((entry_px - current_price) / entry_px * 100)
        
        if pnl_pct <= EMERGENCY_SL_PCT:
            print(f"🚨 緊急損切り: {pnl_pct:.2f}%")
            self.trader.close_position(self.symbol)
            self.risk_manager.update_position_tracking(0, "CLOSE")
        elif pnl_pct >= SECURE_PROFIT_TP_PCT:
            print(f"🎉 緊急利確: {pnl_pct:.2f}%")
            self.trader.close_position(self.symbol)
            self.risk_manager.update_position_tracking(0, "CLOSE")



    def _get_position_summary(self, account_state: dict) -> dict:
        """
        【クリーンアップ版】対象シンボルのポジション情報を一括取得
        - ループ処理を1回に集約
        - 戻り値: サイズ, サイド, PnL, 参入価格, 価値
        """
        # デフォルト値（ポジションなし）
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
            # シンボルが一致し、かつサイズが0でない場合
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
    """
    メイン関数
    """
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    
    # ネットワーク名を判定
    network = os.getenv("NETWORK", "testnet").lower()
    net_display = "MAINNET" if network == "mainnet" else "TESTNET"

     # 環境変数から設定を読み込み
    symbol = os.getenv('TRADING_SYMBOL', 'ETH')
    env_capital = os.getenv('INITIAL_CAPITAL', '1000')
    interval = int(os.getenv('CHECK_INTERVAL', '60'))
    enable_sheets = os.getenv('ENABLE_SHEETS_LOGGING', 'true').lower() == 'true'

    # 資金設定のパース
    try:
        capital = float(env_capital)
    except ValueError:
        print(f"⚠️ エラー: .envのINITIAL_CAPITAL '{env_capital}' が数値ではありません。デフォルトの1000.0を使用します。")
        capital = 1000.0
    
    if mode == 'test':
        print(f"🧪 {net_display} 接続テスト\n")
        trader = HyperliquidSDKTrader()
        
        # 価格取得テスト
        price = trader.get_current_price(symbol) # ETH固定ではなく環境変数を使用
        if price:
            print(f"\n✅ 価格取得成功: ${price:.2f}\n")
        
        # アカウント状態テスト
        trader.print_account_status()
        
        # Risk Managerテスト
        rm = RiskManager(capital)
        rm.print_risk_status()
        
        # Google Sheetsテスト
        try:
            logger = GoogleSheetsLogger()
            print(f"\n✅ Google Sheets接続成功")
            print(f"   URL: {logger.get_spreadsheet_url()}")
        except Exception as e:
            print(f"\n⚠️ Google Sheets接続失敗: {e}")
    
    elif mode == 'status':
        trader = HyperliquidSDKTrader()
        trader.print_account_status()
        
    elif mode == 'buy':
        if len(sys.argv) < 3:
            print(f"使用方法: python main.py buy 0.004")
            return
        
        size = float(sys.argv[2])
        trader = HyperliquidSDKTrader()
        trader.place_order(symbol, is_buy=True, size=size, order_type="market")
        
    elif mode == 'sell':
        if len(sys.argv) < 3:
            print(f"使用方法: python main.py sell 0.004")
            return
        
        size = float(sys.argv[2])
        trader = HyperliquidSDKTrader()
        trader.place_order(symbol, is_buy=False, size=size, order_type="market")
        
    elif mode == 'close':
        trader = HyperliquidSDKTrader()
        trader.close_position(symbol)
        
    elif mode == 'sheets':
        try:
            logger = GoogleSheetsLogger()
            print(f"✅ Google Sheets接続成功")
            print(f"\n📊 スプレッドシートURL:")
            print(f"{logger.get_spreadsheet_url()}\n")
            
            # ✅ 【修正】古い log_trade を log_execution に変更
            logger.log_execution({
                'timestamp': datetime.now(),
                'action': 'BUY',
                'side': 'LONG',
                'size': 0.01,
                'price': 3500.0,
                'fee': 0.035,
                'realized_pnl': 0,
                'balance': 1000.0,
                'reasoning': 'システムテスト(手動実行)'
            })
            
            print("✅ テストデータを記録しました")
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            import traceback
            traceback.print_exc()
        
    else:
        # 自動トレーディング実行
        print(f"\n🚀 {net_display} モードで起動準備中...")
        
        # 実際の残高をチェック
        try:
            temp_trader = HyperliquidSDKTrader()
            account_state = temp_trader.get_user_state()
            
            real_balance = 0.0
            if account_state:
                cross_margin = account_state.get('crossMarginSummary', {})
                margin_summary = account_state.get('marginSummary', {})
                # Perpsの利用可能残高を取得
                real_balance = float(cross_margin.get('totalRawUsd', 0)) or float(margin_summary.get('totalRawUsd', 0))
            
            print(f"💳 ウォレット実残高 (Perps): ${real_balance:.2f}")
            print(f"⚙️ 設定された初期資金: ${capital:.2f}")
            
            if real_balance < capital:
                print(f"⚠️ 警告: 実残高 (${real_balance:.2f}) が設定資金 (${capital:.2f}) を下回っています。")
                print(f"   リスク管理は設定資金 (${capital:.2f}) を基準に計算されます。")
            elif real_balance > capital * 1.5:
                print(f"ℹ️ 情報: 実残高が設定資金より大幅に多いです。リスク管理は設定値(${capital:.2f})に基づいて保守的に行われます。")
                
        except Exception as e:
            print(f"⚠️ 残高チェック時にエラーが発生しました: {e}")
        
        bot = TradingBot(
            symbol=symbol, 
            initial_capital=capital,
            enable_sheets_logging=enable_sheets
        )
        bot.run_trading_loop(interval=interval)

if __name__ == "__main__":
    main()