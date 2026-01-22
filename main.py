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
from ws_monitor import OrderBookMonitor

load_dotenv()

# 緊急損切り・利確設定
EMERGENCY_SL_PCT = float(os.getenv('EMERGENCY_STOP_LOSS', '-2.0')) # デイトレ用にタイトに設定
SECURE_PROFIT_TP_PCT = float(os.getenv('SECURE_TAKE_PROFIT', '4.0'))
MIN_SIGNAL_STRENGTH = int(os.getenv('MIN_SIGNAL_STRENGTH', '45'))

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

        # トレードの文脈を保存する変数
        self.trade_context = {
            'entry_price': 0.0,
            'entry_reason': '',
            'size': 0.0,
            'side': 'NONE'
        }

        # 機械学習予測器
        self.ml_predictor = MLPredictor(symbol=symbol)
        # 15分足ベースで学習するように設定
        self.online_learner = OnlineLearner(symbol=symbol, timeframe=MAIN_TIMEFRAME, retrain_interval_hours=24)
        print(f"🤖 機械学習予測システム: 有効 (Timeframe: {MAIN_TIMEFRAME})")
        print(f"   モデル状態: {self.ml_predictor.lgb_model is not None or self.ml_predictor.lstm_model is not None}")
        
        # Google Sheetsロガー初期化
        self.sheets_logger = None
        from collections import deque
        self.prediction_history = deque() 
        if self.enable_sheets_logging:
            try:
                self.sheets_logger = GoogleSheetsLogger()
                print(f"📊 Google Sheetsログ記録: 有効")
            except Exception as e:
                print(f"⚠️ Google Sheetsログ記録を無効化: {e}")
                self.enable_sheets_logging = False

        # 監視システムの起動
        self.ws_monitor = OrderBookMonitor(symbol=symbol)
        self.ws_monitor.start() # ここでスパイが出動
        time.sleep(2) # 接続待ち

        # OI（建玉）の変化を追跡するための変数
        self.last_oi = 0.0
        
        print("\n" + "="*70)
        print(f"🚀 Hyperliquid {self.bot_name} Bot (DayTrade Logic)")
        print("="*70)

    
    def get_ml_decision(self, market_analysis: dict, account_state: dict, structure_data: dict) -> dict:
        """
        【修正: デイトレ・高頻度版】
        - 閾値を下げてエントリー回数を増やす
        - AIの確信度が低くても、板情報(Imbalance)が強ければエントリーする
        """
        try:
            # === ステップ1: データ取得 (15分足) ===
            df_main = self.market_data.get_ohlcv(MAIN_TIMEFRAME, limit=200)
            
            # 板情報の偏りを取得 (プラスなら買い圧、マイナスなら売り圧)
            fast_imbalance = self.ws_monitor.get_latest_imbalance()
            print(f"⚡ 高速板情報: {fast_imbalance:.2f}")

            # OI変化率を取り出す
            oi_delta = structure_data.get('oi_delta_pct', 0.0)
            
            # === ステップ2: ML予測実行 ===
            ml_result = self.ml_predictor.predict(df_main)
            
            if ml_result.get('model_used') == 'NONE':
                return {'action': 'HOLD', 'side': 'NONE', 'confidence': 0, 'reasoning': 'モデル未学習'}
            
            # === ステップ3: 確率分布の解析 ===
            up_prob = ml_result['up_prob']
            down_prob = ml_result['down_prob']
            confidence = ml_result['confidence'] # これは max(up, down) * 100
            
            # 既存ポジション確認
            existing_side = None
            if account_state and 'assetPositions' in account_state:
                for pos in account_state['assetPositions']:
                    p = pos.get('position', {})
                    if p.get('coin') == self.symbol and float(p.get('szi', 0)) != 0:
                        existing_side = 'LONG' if float(p.get('szi', 0)) > 0 else 'SHORT'
                        break
            
            # --- ★変更: デイトレ用の閾値設定 ---
            # 基本エントリーラインを大幅に下げる
            BASE_THRESHOLD = 0.53  
            # 撤退ラインも早める (回転率重視)
            CLOSE_THRESHOLD = 0.55 

            action = 'HOLD'
            side = 'NONE'
            reasoning = f"Wait: Up({up_prob:.2f}) Down({down_prob:.2f})"

            # 指標取得
            indicators = market_analysis.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            current_price = market_analysis.get('price', 0)
            sma_50 = indicators.get('sma_50', current_price)

            
            # 1. 板情報による補正 (Fast Imbalance Boost) ===
            adjusted_up_prob = up_prob
            adjusted_down_prob = down_prob

            # 閾値設定: 0.3 (全体の30%以上の偏り) があればAIを後押しする
            BOOST_VAL = 0.05 # 5%の確率加算

            if fast_imbalance > 0.3: # 買い板が強い
                adjusted_up_prob += BOOST_VAL 
                reasoning += f" [板:買い有利({fast_imbalance:.2f})]"
            elif fast_imbalance < -0.3: # 売り板が強い
                adjusted_down_prob += BOOST_VAL 
                reasoning += f" [板:売り有利({fast_imbalance:.2f})]"

            # 2. OIフィルター & ブースト
            # OIが減少している(ショートカバー等の手仕舞い)場合、トレンドフォローの確率を下げる（ダマシ回避）
            if oi_delta < -0.05: # -0.05%以上減少（15秒間でこれは大きな動き）
                adjusted_up_prob -= 0.05
                adjusted_down_prob -= 0.05
                reasoning += f" [OI減:手仕舞い警戒]"
            
            # OIが急増している（本気の資金流入）場合、方向感を後押し
            elif oi_delta > 0.05:
                # どちらかの確率が既に高いなら、それをさらに後押し
                if adjusted_up_prob > adjusted_down_prob:
                    adjusted_up_prob += 0.03
                    reasoning += f" [OI増:追随]"
                elif adjusted_down_prob > adjusted_up_prob:
                    adjusted_down_prob += 0.03
                    reasoning += f" [OI増:追随]"

            # 補正後の自信度を再計算
            adjusted_confidence = max(adjusted_up_prob, adjusted_down_prob) * 100

            if existing_side:
                # === 決済ロジック (早めに逃げる) ===
                # 逆行確率が50%を超えたら即撤退 (含み損を拡大させない)
                if existing_side == 'LONG' and down_prob > CLOSE_THRESHOLD:
                    action = 'CLOSE'
                    reasoning = f'LONG撤退: 下落予測 ({down_prob*100:.1f}%)'
                elif existing_side == 'SHORT' and up_prob > CLOSE_THRESHOLD:
                    action = 'CLOSE'
                    reasoning = f'SHORT撤退: 上昇予測 ({up_prob*100:.1f}%)'
                
                # 2時間経過で撤退 (回転率アップ)
                if self.last_entry_time and (datetime.now() - self.last_entry_time).total_seconds() > 2 * 3600:
                    action = 'CLOSE'
                    reasoning = 'TimeExit: 2時間経過'

            else:
                # === 新規エントリーロジック (ハイブリッド判定) ===
                
                # --- 買い判定 ---
                # 板情報(imbalance)による逆張り許可を削除。SMA50のトレンドフィルターを厳格化。
                is_trend_ok_buy = (current_price > sma_50)
                
                if (adjusted_up_prob >= BASE_THRESHOLD and 
                    adjusted_up_prob > adjusted_down_prob and 
                    rsi < 70 and 
                    is_trend_ok_buy):
                    
                    action = 'BUY'
                    side = 'LONG'
                    reasoning = f'BUY: 予測{adjusted_up_prob*100:.1f}%'
                
                    # --- 売り判定 ---
                    is_trend_ok_sell = (current_price < sma_50)
                
                elif (adjusted_down_prob >= BASE_THRESHOLD and 
                      adjusted_down_prob > adjusted_up_prob and 
                      rsi > 30 and 
                      is_trend_ok_sell):
                      
                    action = 'SELL'
                    side = 'SHORT'
                    reasoning = f'SELL: 予測{adjusted_down_prob*100:.1f}%'
            
            # === 動的リスクパラメータ (回転率重視) ===
            volatility = market_analysis.get('volatility', 2.0)
            
            # 利益1.5%〜3.0%でサクサク利確する
            if volatility > 3.0: 
                sl_pct, tp_pct = 2.0, 3.5 
            else: 
                sl_pct, tp_pct = 1.0, 2.0 # 狭く設定してヒット率を上げる

            # 期待値計算
            win_prob = adjusted_up_prob if action == 'BUY' else adjusted_down_prob if action == 'SELL' else 0.0
            if action in ['BUY', 'SELL']:
                expected_value_r = (win_prob * tp_pct) - ((1 - win_prob) * sl_pct)
            else:
                expected_value_r = 0

            # ログ表示
            print(f"\n🤖 ML判断詳細 (Boosted):")
            print(f"   Model: {ml_result['model_used']}")
            print(f"   Raw Prob: Up {up_prob*100:.1f}% | Down {down_prob*100:.1f}%")
            print(f"   Adj Prob: Up {adjusted_up_prob*100:.1f}% | Down {adjusted_down_prob*100:.1f}%")
            print(f"   Fast Imbalance: {fast_imbalance:.2f}")
            print(f"   Action: {action} (Conf: {adjusted_confidence:.1f})")

            return {
                'action': action,
                'side': side,
                'confidence': adjusted_confidence, # 補正後の自信度を返す
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
                    'volatility': signal_data.get('volatility', 0),
                    'price_diff': signal_data.get('price_diff', '-'),
                    'prediction_result': signal_data.get('prediction_result', '-')
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

        # === 1. EV/RRチェック (手数料考慮版) ===
        ev = float(decision.get('expected_value_r', 0))
        rr_ratio = float(decision.get('risk_reward_ratio', 0))
        
        # 手数料負けガード (Taker往復 0.07% + バッファ)
        ESTIMATED_COST_PCT = 0.1
        net_ev = ev - ESTIMATED_COST_PCT

        if action in ['BUY', 'SELL']:
            if net_ev <= 0.3: 
                print(f"🛑 取引拒否: 手数料負けリスク (Net EV: {net_ev:.2f}%)")
                return
            if rr_ratio < 1.2:
                print(f"🛑 取引拒否: リスクリワード比不足 (RR: {rr_ratio:.2f})")
                return
        
        # === 2. アカウント情報・既存ポジション一括取得 ===
        cross_margin = account_state.get('crossMarginSummary', {}) if account_state else {}
        margin_summary = account_state.get('marginSummary', {}) if account_state else {}
        account_value = float(cross_margin.get('accountValue', 0)) or float(margin_summary.get('accountValue', 0))
        available_balance = float(cross_margin.get('totalRawUsd', 0)) or float(margin_summary.get('totalRawUsd', 0))
        
        self.risk_manager.current_capital = account_value
        
        # ★重要: 再起動時などのために、ここでのポジション情報を確保しておく
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
        size = 0.0
        risk_level = "CLOSE"
        reasoning = decision.get('reasoning')
        order_value = 0.0

        if action != 'CLOSE':
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
            reasoning = position_result['reasoning'] # エントリー理由を更新
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
                # --- 詳細なトレード結果の計算と記録 ---
                exit_price = current_price
                
                # ★修正: メモリにない場合(再起動後など)は、APIから取得したpos_dataを使う
                if self.trade_context['size'] > 0:
                    entry_price = self.trade_context['entry_price']
                    size_closed = self.trade_context['size']
                    side_closed = self.trade_context['side']
                    entry_reason = self.trade_context['entry_reason']
                else:
                    # 救済措置: メモリが消えていてもAPI情報から計算
                    entry_price = pos_data['entry_price']
                    size_closed = pos_data['size']
                    side_closed = pos_data['side']
                    entry_reason = "Unknown (Bot Restarted)"

                # 損益計算
                if side_closed == 'LONG':
                    raw_pnl = (exit_price - entry_price) * size_closed
                else: # SHORT
                    raw_pnl = (entry_price - exit_price) * size_closed
                
                # 手数料推定 (往復 0.07%)
                fee_cost = (entry_price * size_closed * 0.00035) + (exit_price * size_closed * 0.00035)
                net_pnl = raw_pnl - fee_cost
                
                # 経過時間
                if self.last_entry_time:
                    duration = datetime.now() - self.last_entry_time
                else:
                    duration = timedelta(0)
                
                # ログ送信
                self.sheets_logger.log_trade_result({
                    'exit_time': datetime.now(),
                    'symbol': self.symbol,
                    'side': side_closed,
                    'size': size_closed,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': round(net_pnl, 2),
                    'duration': str(duration).split('.')[0],
                    'entry_reason': entry_reason,
                    'exit_reason': decision.get('reasoning')
                })
                
                print(f"📝 トレード結果記録: PnL ${net_pnl:.2f}")

                # コンテキストのリセット
                self.last_entry_time = None
                self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE'}
                self.risk_manager.update_position_tracking(0, "CLOSE")

        else:
            # --- エントリー注文 ---
            # SL/TP価格計算 (ログ表示用)
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
                # ★修正: エントリー成功時のみコンテキストを更新する（CLOSEのロジックと分離）
                self.trade_context = {
                    'entry_price': current_price,
                    'entry_reason': reasoning,
                    'size': size,
                    'side': side
                }
                self.last_entry_time = datetime.now()
                self.risk_manager.update_position_tracking(order_value, "ADD")
            else:
                print("❌ 取引失敗")

        # === 9. Google Sheetsログ記録 (Executions/AI/Equity) ===
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
                'realized_pnl': 0, # ここは未確定分
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
                'model_used': decision.get('reasoning', '').split('|')[-1].strip(),
                'price_diff': decision.get('price_diff', '-'),
                'prediction_result': decision.get('prediction_result', '-')
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
        自動取引ループ (改良版: トレード品質格付け判定付き)
        """
        self.running = True
        self.online_learner.start_background_learning()
        
        print(f"\n🚀 自動トレーディング開始")
        print(f"   判断間隔: {interval}秒")
        print(f"   メイン時間軸: {MAIN_TIMEFRAME}")
        
        try:
            last_ai_check_time = 0
            fast_interval = 1 

            last_ai_state = {
                'price': None,
                'up_prob': 0,
                'down_prob': 0,
                'action': 'HOLD'
            }
            
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
                    fast_imbalance = self.ws_monitor.get_latest_imbalance()

                    # 3. OIの取得と変化率計算
                    # WebSocket経由で高速・確実にOIを取得
                    current_oi = self.ws_monitor.get_latest_oi()
                    if current_oi == 0:
                        current_oi = self.market_data.get_open_interest()

                    oi_delta_pct = 0.0
                    if self.last_oi > 0:
                        oi_delta_pct = ((current_oi - self.last_oi) / self.last_oi) * 100
                    
                    # 変化がない(0.0)場合は、取得失敗等の可能性もあるため更新しない手もありだが、
                    # ここでは常に最新を正とする
                    if current_oi > 0:
                        self.last_oi = current_oi
                    
                    if analysis:
                        volatility = analysis.get('volatility', 0)
                        print(f"   Vol: {volatility:.2f}% | Imb: {fast_imbalance:.2f} | OI: {current_oi:.2f} | OI Δ: {oi_delta_pct:+.4f}%")

                        # 設定した閾値（.envのLOW_VOLATILITY_THRESHOLD、デフォルト1.5）未満ならスキップ
                        # ここでは安全のためハードコード気味に 1.0% 未満は絶対停止とする例
                        MIN_VOLATILITY_LIMIT = 1.0 
                        
                        if volatility < MIN_VOLATILITY_LIMIT:
                            print(f"💤 低ボラティリティのため待機 (Vol: {volatility:.2f}% < {MIN_VOLATILITY_LIMIT}%)")
                            last_ai_check_time = current_time # 時間は更新して、次のインターバルまで寝る
                            time.sleep(fast_interval)
                            continue
                        
                        # 3. ML判断を実行
                        structure['oi_delta_pct'] = oi_delta_pct
                        decision = self.get_ml_decision(analysis, account_state, structure)
                        
                        if decision:
                            action = decision.get('action', 'HOLD')
                            confidence = decision.get('confidence', 0)
                            up_prob = decision['ml_probabilities']['up']
                            down_prob = decision['ml_probabilities']['down']
                            
                            # === 変数初期化 ===
                            prediction_result = "⏳ 判定待ち" 
                            price_diff_str = "-"

                            # 1. 現在の予測を履歴に追加
                            # ★改善: 「その時の自信度(confidence)」も一緒に保存する
                            self.prediction_history.append({
                                'timestamp': current_time,
                                'price': current_price,
                                'up_prob': up_prob,
                                'down_prob': down_prob,
                                'confidence': confidence # 追加
                            })

                            # 2. 15分以上前のデータを探して検証
                            while len(self.prediction_history) > 0:
                                old_data = self.prediction_history[0]
                                time_diff = current_time - old_data['timestamp']
                                
                                if time_diff < 900: # 15分未満なら終了
                                    break
                                
                                target_data = self.prediction_history.popleft()
                                
                                # --- 答え合わせロジック (格付け機能付き) ---
                                past_price = target_data['price']
                                past_conf = target_data.get('confidence', 0)
                                price_change = current_price - past_price
                                sign = "+" if price_change > 0 else ""
                                price_diff_str = f"{sign}{price_change:.2f}" 
                                
                                past_up = target_data['up_prob']
                                past_down = target_data['down_prob']
                                
                                # トレード対象になるレベルだったか？ (Conf >= 35)
                                is_trade_level = (past_conf >= 35)

                                result_label = "⚪️ Draw"
                                
                                # AI予測: 上昇
                                if past_up > past_down:
                                    if price_change > 0:
                                        # 的中
                                        if is_trade_level: result_label = f"🏆 トレード勝利 (Conf:{past_conf:.0f})"
                                        else: result_label = "✅ 方向正解 (見送り)"
                                    else:
                                        # ハズレ
                                        if is_trade_level: result_label = f"💀 トレード敗北 (Conf:{past_conf:.0f})"
                                        else: result_label = "❌ 方向不正解"
                                
                                # AI予測: 下落
                                elif past_down > past_up:
                                    if price_change < 0:
                                        # 的中
                                        if is_trade_level: result_label = f"🏆 トレード勝利 (Conf:{past_conf:.0f})"
                                        else: result_label = "✅ 方向正解 (見送り)"
                                    else:
                                        # ハズレ
                                        if is_trade_level: result_label = f"💀 トレード敗北 (Conf:{past_conf:.0f})"
                                        else: result_label = "❌ 方向不正解"
                                
                                prediction_result = result_label
                                break

                            # decisionに結果を格納
                            decision['price_diff'] = price_diff_str
                            decision['prediction_result'] = prediction_result
                            
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
                                'model_used': decision.get('reasoning', '').split('|')[-1].strip(),
                                'price_diff': price_diff_str,
                                'prediction_result': prediction_result
                            }
                            
                            # 5. 取引実行 または ログ記録のみ
                            if action == "CLOSE":
                                self.execute_trade(decision, current_price, account_state, analysis)
                            
                            elif action in ['BUY', 'SELL']:
                                if confidence >= 50:
                                    self.execute_trade(decision, current_price, account_state, analysis)
                                else:
                                    print(f"⏸️ 信頼度不足で見送り ({confidence:.1f}%)")
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
    interval = int(os.getenv('CHECK_INTERVAL', '15'))
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