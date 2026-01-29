# main.py (予測判定刷新版: 理由可視化特化)
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

# トレード実行/クローズの確率閾値設定
BASE_THRESHOLD  = float(os.getenv('BASE_THRESHOLD', '0.47'))
CLOSE_THRESHOLD = float(os.getenv('CLOSE_THRESHOLD', '0.51'))

# 緊急損切り・利確設定
EMERGENCY_SL_PCT = float(os.getenv('EMERGENCY_STOP_LOSS', '-2.0')) # デイトレ用にタイトに設定
SECURE_PROFIT_TP_PCT = float(os.getenv('SECURE_TAKE_PROFIT', '4.0'))

# 時間軸設定
MAIN_TIMEFRAME = os.getenv('MAIN_TIMEFRAME', '15m')  # デイトレードの主軸
TREND_TIMEFRAME = os.getenv('TREND_TIMEFRAME', '1h') # 環境認識用

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
            'side': 'NONE',
            'sl_percent': None,
            'tp_percent': None 
        }

        # 状態保存ファイルのパス
        self.state_file = "bot_state.json"

        # 機械学習予測器
        self.ml_predictor = MLPredictor(symbol=symbol)
        # 15分足ベースで学習するように設定
        self.online_learner = OnlineLearner(
            symbol=symbol, 
            timeframe=MAIN_TIMEFRAME, 
            retrain_interval_hours=4 
        )
        print(f"🤖 機械学習予測システム: 有効 (Timeframe: {MAIN_TIMEFRAME})")
        print(f"   モデル状態: {self.ml_predictor.lgb_model is not None or self.ml_predictor.lstm_model is not None}")
        
        # Google Sheetsロガー初期化
        self.sheets_logger = None
        # ★履歴管理用のdequeは不要になったため削除
        
        if self.enable_sheets_logging:
            try:
                self.sheets_logger = GoogleSheetsLogger()
                print(f"📊 Google Sheetsログ記録: 有効")
            except Exception as e:
                print(f"⚠️ Google Sheetsログ記録を無効化: {e}")
                self.enable_sheets_logging = False

        # 監視システムの起動
        self.ws_monitor = OrderBookMonitor(symbol=symbol)
        self.ws_monitor.start() 
        time.sleep(2) # 接続待ち

        # OI（建玉）の変化を追跡するための変数
        self.last_oi = 0.0
        
        # 起動時に前回の状態を復元する
        self._load_bot_state()

        print("\n" + "="*70)
        print(f"🚀 Hyperliquid {self.bot_name} Bot (DayTrade Reasoning Mode)")
        print("="*70)



    # -----------------------------------------------------------
    # 状態の保存と読み込み
    # -----------------------------------------------------------
    def _save_bot_state(self):
        """現在のトレード状態をJSONファイルに保存"""
        try:
            data = {
                'last_entry_time': self.last_entry_time.isoformat() if self.last_entry_time else None,
                'trade_context': self.trade_context
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=4)
            # print("💾 Bot状態を保存しました")
        except Exception as e:
            print(f"⚠️ 状態保存エラー: {e}")
    


    def _load_bot_state(self):
        """JSONファイルからトレード状態を復元"""
        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            
            # 時刻の復元
            if data.get('last_entry_time'):
                self.last_entry_time = datetime.fromisoformat(data['last_entry_time'])
            
            # コンテキストの復元
            if data.get('trade_context'):
                self.trade_context = data['trade_context']
                
            # 実際のポジションがあるか確認し、なければリセット
            # (ファイルには残っているが、手動決済などで消えている場合の整合性チェック)
            account_state = self.trader.get_user_state()
            pos_data = self._get_position_summary(account_state)
            
            if not pos_data['found']:
                # ポジションがないのにデータが残っていたらクリア
                if self.last_entry_time is not None:
                    print("⚠️ ポジション不整合を検知: 状態をリセットします")
                    self.last_entry_time = None
                    self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE'}
                    self._save_bot_state()
            
        except Exception as e:
            print(f"⚠️ 状態復元エラー: {e}")


    
    def get_ml_decision(self, market_analysis: dict, account_state: dict, structure_data: dict) -> dict:
        """
        【判定ロジック】
        Why（理由）を明確にするためのロジック構築
        """
        try:
            # === ステップ1: データ取得 (15分足) ===
            df_main = self.market_data.get_ohlcv(MAIN_TIMEFRAME, limit=200)
            
            # 板情報の偏りを取得
            fast_imbalance = self.ws_monitor.get_latest_imbalance()
            print(f"⚡ 高速板情報: {fast_imbalance:.2f}")

            # OI変化率を取り出す
            oi_delta = structure_data.get('oi_delta_pct', 0.0)
            
            # === ステップ2: ML予測実行 ===
            ml_result = self.ml_predictor.predict(df_main, extra_features=structure_data)
            
            # 予測不能時の早期リターン
            if ml_result.get('model_used') == 'NONE':
                return {
                    'action': 'HOLD', 'side': 'NONE', 'confidence': 0, 
                    'reasoning': 'Wait: モデル未学習', 'ml_probabilities': {'up': 0.0, 'down': 0.0}
                }
            
            # === ステップ3: 確率分布の解析 ===
            up_prob = ml_result['up_prob']
            down_prob = ml_result['down_prob']
            
            # 既存ポジション確認
            existing_side = None
            if account_state and 'assetPositions' in account_state:
                for pos in account_state['assetPositions']:
                    p = pos.get('position', {})
                    if p.get('coin') == self.symbol and float(p.get('szi', 0)) != 0:
                        existing_side = 'LONG' if float(p.get('szi', 0)) > 0 else 'SHORT'
                        break

            # 初期状態の設定
            action = 'HOLD'
            side = 'NONE'
            if existing_side:
                reasoning = f"Hold: {existing_side}継続"
            else:
                reasoning = f"Wait: 様子見"

            # 指標取得
            indicators = market_analysis.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            current_price = market_analysis.get('price', 0)

            # === 1. 確率補正 (OIフィルター & ブースト) ===
            adjusted_up_prob = up_prob
            adjusted_down_prob = down_prob

            if oi_delta < -0.05: 
                adjusted_up_prob -= 0.05
                adjusted_down_prob -= 0.05
                reasoning += f" [OI減]"
            elif oi_delta > 0.05:
                if adjusted_up_prob > adjusted_down_prob:
                    adjusted_up_prob += 0.03
                elif adjusted_down_prob > adjusted_up_prob:
                    adjusted_down_prob += 0.03

            # スコア補正
            signal_score = market_analysis.get('signal_strength', 50)
            score_adjust = (signal_score - 50) * 0.001 
            adjusted_up_prob += score_adjust
            adjusted_down_prob -= score_adjust
            
            # 補正後の自信度
            adjusted_confidence = max(adjusted_up_prob, adjusted_down_prob) * 100

            # 動的閾値計算 (共通)
            raw_adj        = fast_imbalance * 0.08
            threshold_adj  = max(min(raw_adj, 0.08), -0.08)
            buy_threshold  = BASE_THRESHOLD - threshold_adj
            sell_threshold = BASE_THRESHOLD + threshold_adj


            if existing_side:
                # === 決済ロジック (イベント駆動型: 15分/30分チェック) ===
                
                # 経過時間の計算
                elapsed_minutes = (datetime.now() - self.last_entry_time).total_seconds() / 60 if self.last_entry_time else 0
                
                # 現在のPnL(%)を計算
                current_pnl_pct = 0.0
                pos_summary = self._get_position_summary(account_state)
                if pos_summary['found']:
                    entry_px = pos_summary['entry_price']
                    if existing_side == 'LONG':
                        current_pnl_pct = (current_price - entry_px) / entry_px * 100
                    else:
                        current_pnl_pct = (entry_px - current_price) / entry_px * 100

                # --- 【フェーズ 4】 45分以降: タイムアップ ---
                if elapsed_minutes > 45:
                    action = 'CLOSE'
                    reasoning = f'CLOSE: 45分経過 (タイムリミット)'

                # --- 【フェーズ 3】 30分経過時のワンショット判定 ---
                elif elapsed_minutes >= 30:
                    # まだ30分チェックを行っていない場合のみ実行
                    if not self.trade_context.get('check_30m_done', False):
                        print(f"⏰ 30分経過: 継続審査を実行中...")
                        
                        # フラグを立てて保存（二度と呼ばれないようにする）
                        self.trade_context['check_30m_done'] = True
                        # 30分経っているので15分フラグもTrueにしておく（念のため）
                        self.trade_context['check_15m_done'] = True
                        self._save_bot_state()

                        # 判定: 厳格な閾値 (+0.02) をクリアしているか？
                        strict_buy_th = buy_threshold + 0.02
                        strict_sell_th = sell_threshold + 0.02
                        
                        if existing_side == 'LONG' and adjusted_up_prob < strict_buy_th:
                            action = 'CLOSE'
                            reasoning = f'CLOSE: 30分審査落ち (Up:{adjusted_up_prob:.2f} < {strict_buy_th:.2f})'
                        elif existing_side == 'SHORT' and adjusted_down_prob < strict_sell_th:
                            action = 'CLOSE'
                            reasoning = f'CLOSE: 30分審査落ち (Down:{adjusted_down_prob:.2f} < {strict_sell_th:.2f})'
                        else:
                            print("✅ 30分審査通過: ホールド継続")
                            reasoning += " [30m審査済]"
                    
                    else:
                        # 審査通過後のホールド期間
                        reasoning += " [30m通過]"

                # --- 【フェーズ 2】 15分経過時のワンショット判定 ---
                elif elapsed_minutes >= 15:
                    # まだ15分チェックを行っていない場合のみ実行
                    if not self.trade_context.get('check_15m_done', False):
                        print(f"⏰ 15分経過: 継続審査を実行中...")
                        
                        # フラグを立てる
                        self.trade_context['check_15m_done'] = True
                        self._save_bot_state()

                        # 判定A: 含み損なら即撤退
                        if current_pnl_pct < 0:
                            action = 'CLOSE'
                            reasoning = f'CLOSE: 15分審査落ち (含み損 {current_pnl_pct:.2f}%)'
                        
                        # 判定B: 予測確率がエントリー基準を下回っていたら撤退
                        elif existing_side == 'LONG' and adjusted_up_prob < buy_threshold:
                            action = 'CLOSE'
                            reasoning = f'CLOSE: 15分審査落ち (Up:{adjusted_up_prob:.2f} < {buy_threshold:.2f})'
                        elif existing_side == 'SHORT' and adjusted_down_prob < sell_threshold:
                            action = 'CLOSE'
                            reasoning = f'CLOSE: 15分審査落ち (Down:{adjusted_down_prob:.2f} < {sell_threshold:.2f})'
                        else:
                            print("✅ 15分審査通過: ホールド継続")
                            reasoning += " [15m審査済]"
                    
                    else:
                        # 審査通過後のホールド期間
                        reasoning += " [15m通過]"

                # --- 【フェーズ 1 & 全期間共通】 緊急・逆行監視 ---
                else:
                    # 0-15分、または各審査通過後の期間
                    
                    # RSI過熱感での利確 (常時有効)
                    if existing_side == 'LONG' and rsi > 70:
                        action = 'CLOSE'
                        reasoning = f'CLOSE: RSI過熱 ({rsi:.1f})'
                    elif existing_side == 'SHORT' and rsi < 30:
                        action = 'CLOSE'
                        reasoning = f'CLOSE: RSI売られすぎ ({rsi:.1f})'
                    
                    # 完全なドテン（逆シグナル）が出た場合は流石に逃げる
                    elif existing_side == 'LONG' and down_prob > CLOSE_THRESHOLD + 0.05: # 少し余裕を持たせる
                        action = 'CLOSE'
                        reasoning = f'CLOSE: 強い逆行シグナル (Down:{down_prob*100:.1f}%)'
                    elif existing_side == 'SHORT' and up_prob > CLOSE_THRESHOLD + 0.05:
                        action = 'CLOSE'
                        reasoning = f'CLOSE: 強い逆行シグナル (Up:{up_prob*100:.1f}%)'

                    else:
                        reasoning += f" | ⏳{int(elapsed_minutes)}分 (PnL:{current_pnl_pct:+.2f}%)"

            else:
                # === 新規エントリーロジック (閾値計算は上で実施済み) ===
                if (adjusted_up_prob >= buy_threshold and 
                    adjusted_up_prob > adjusted_down_prob and 
                    rsi < 75): 
                    
                    action = 'BUY'
                    side = 'LONG'
                    reasoning = f'BUY: 予測{adjusted_up_prob*100:.1f}% > 閾値{buy_threshold*100:.1f}%'
                
                elif (adjusted_down_prob >= sell_threshold and 
                      adjusted_down_prob > adjusted_up_prob and 
                      rsi > 25):
                      
                    action = 'SELL'
                    side = 'SHORT'
                    reasoning = f'SELL: 予測{adjusted_down_prob*100:.1f}% > 閾値{sell_threshold*100:.1f}%'
                
                else:
                    # Wait理由
                    max_p = max(adjusted_up_prob, adjusted_down_prob)
                    target_th = buy_threshold if adjusted_up_prob > adjusted_down_prob else sell_threshold
                    reasoning += f" (Prob:{max_p:.2f} < Th:{target_th:.2f})"
            
            # === 動的リスクパラメータ ===
            volatility = market_analysis.get('volatility', 2.0)
            if volatility > 3.0: 
                sl_pct, tp_pct = 2.0, 3.5 
            else: 
                sl_pct, tp_pct = 1.0, 2.0

            # 期待値計算
            win_prob = adjusted_up_prob if action == 'BUY' else adjusted_down_prob if action == 'SELL' else 0.0
            if action in ['BUY', 'SELL']:
                expected_value_r = (win_prob * tp_pct) - ((1 - win_prob) * sl_pct)
            else:
                expected_value_r = 0

            # ログ表示
            print(f"\n🤖 ML判断詳細 (Boosted):")
            print(f"   Model: {ml_result['model_used']}")
            print(f"   Action: {action} (Conf: {adjusted_confidence:.1f})")
            print(f"   Reason: {reasoning}")

            return {
                'action': action,
                'side': side,
                'confidence': adjusted_confidence,
                'expected_value_r': expected_value_r,
                'risk_reward_ratio': tp_pct / sl_pct,
                'stop_loss_percent': sl_pct,
                'take_profit_percent': tp_pct,
                'reasoning': f"{reasoning} | {ml_result['model_used']}",
                'ml_probabilities': {'up': up_prob, 'down': down_prob},
                'filter_reason': ml_result.get('filter_reason')
            }
            
        except Exception as e:
            print(f"⚠️ ML判断エラー: {e}")
            import traceback
            traceback.print_exc()
            return {
                'action': 'HOLD', 'side': 'NONE', 'confidence': 0, 
                'reasoning': f'Error: {str(e)}', 'ml_probabilities': {'up': 0.0, 'down': 0.0}
            }
    

    
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

    def _log_cancel_reason(self, decision, current_price, analysis, reason_text):
        """
        ★追加: トレード拒否（キャンセル）時の理由をログに記録するヘルパー
        """
        atr_pct = (analysis.get('indicators', {}).get('atr', 0) / current_price * 100) if current_price > 0 else 0
        
        signal_log = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'action': 'WAIT', # キャンセルなのでWAIT扱い
            'confidence': decision.get('confidence', 0),
            'ml_probabilities': decision.get('ml_probabilities', {}),
            'price': current_price,
            'volatility': atr_pct,
            'rsi': analysis.get('indicators', {}).get('rsi', 0),
            'market_regime': decision.get('market_regime', 'NORMAL'),
            'model_used': decision.get('reasoning', '').split('|')[-1].strip(),
            'price_diff': '-',
            'prediction_result': f"⛔ {reason_text}" # 理由をここに明記
        }
        self.log_to_sheets(signal_data=signal_log)


    def execute_trade(self, decision: dict, current_price: float, account_state: dict, analysis: dict):
        """
        実際の取引を実行してGoogle Sheetsに記録
        """
        action = decision.get('action')

        # === 1. EV/RRチェック (手数料考慮版・緩和) ===
        ev = float(decision.get('expected_value_r', 0))
        rr_ratio = float(decision.get('risk_reward_ratio', 0))
        
        # 手数料負けガード (Taker往復 0.07% + バッファ)
        ESTIMATED_COST_PCT = 0.1
        net_ev = ev - ESTIMATED_COST_PCT

        if action in ['BUY', 'SELL']:
            # 【緩和】0.3% -> 0.05%
            if net_ev <= 0.05: 
                reason = f"EV不足(Net:{net_ev:.2f}%)"
                print(f"🛑 取引拒否: {reason}")
                self._log_cancel_reason(decision, current_price, analysis, reason)
                return
            # 【緩和】1.2 -> 0.8
            if rr_ratio < 0.8:
                reason = f"RR不足({rr_ratio:.2f})"
                print(f"🛑 取引拒否: {reason}")
                self._log_cancel_reason(decision, current_price, analysis, reason)
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
            reason = "日次損失限度到達"
            print(f"🛑 {reason}")
            self._log_cancel_reason(decision, current_price, analysis, reason)
            return
        
        # === 4. AI自信度を取得 ===
        confidence = float(decision.get('confidence', 0))
        
        # === 5. 追加ポジション可否判定 (CLOSE以外) ===
        if action != 'CLOSE' and existing_position_value > 0:
            if not self.risk_manager.should_add_position(confidence, existing_position_value):
                reason = "既存Posあり追加不可"
                print(f"⚠️ {reason}")
                self._log_cancel_reason(decision, current_price, analysis, reason)
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
        ai_forecast_info = ""

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

            # 予想変動幅と期待利益の計算
            predicted_change = float(decision.get('predicted_change', 0.0))
            
            # AIが予測する変動方向を考慮 (BUYならプラス、SELLならマイナスの変動幅を期待)
            if side == 'LONG':
                target_change_pct = abs(predicted_change) if predicted_change != 0 else 0.5 
            else:
                target_change_pct = -abs(predicted_change) if predicted_change != 0 else -0.5

            expected_price = current_price * (1 + target_change_pct / 100) # 予想到達価格
            expected_profit = abs(expected_price - current_price) * size # 予想利益 
            
            size        = position_result['size']
            risk_level  = position_result['risk_level']
            reasoning   = position_result['reasoning']
            order_value = position_result['position_value']
            
            print(f"\n✅ 計算結果:")
            print(f"   ポジションサイズ: {size:.4f} ETH")
            print(f"   ポジション金額: ${order_value:.2f}")
            print(f"   リスクレベル: {risk_level}")
            print(f"🔮 AI価格予想:")
            print(f"   予想変動: {target_change_pct:+.2f}%")
            print(f"   目標価格: ${expected_price:.2f}")
            print(f"   期待利益: ${expected_profit:.2f} (手数料別)")
            print(f"{'='*70}\n")
            
            if size == 0:
                reason = "サイズ計算結果0"
                print(f"⚠️ {reason}")
                self._log_cancel_reason(decision, current_price, analysis, reason)
                return
            
            # 予想変動幅と期待利益の計算 & 表示
            predicted_change = float(decision.get('predicted_change', 0.0))
            if side == 'LONG':
                target_change_pct = abs(predicted_change) if predicted_change != 0 else 0.5
            else:
                target_change_pct = -abs(predicted_change) if predicted_change != 0 else -0.5

            expected_price = current_price * (1 + target_change_pct / 100)
            expected_profit = abs(expected_price - current_price) * size
            print(f"🔮 AI価格予想:")
            print(f"   予想変動: {target_change_pct:+.2f}%")
            print(f"   目標価格: ${expected_price:.2f}")
            print(f"   期待利益: ${expected_profit:.2f} (手数料別)")
            print(f"{'='*70}\n")

            # 例: " | 🔮予:+0.45% 🎯$3013 💰$2.25"
            ai_forecast_info = f" | 🔮予:{target_change_pct:+.2f}% 🎯${expected_price:.0f} 💰${expected_profit:.2f}"

        # === 8. 注文実行 ===
        trade_success = False
        estimated_fee = 0.0

        if action == 'CLOSE':
            print(f"📉 ポジション決済実行...")
            result = self.trader.close_position(self.symbol)
            trade_success = result and result.get('status') == 'ok'
            
            if trade_success:
                exit_price = current_price
                if self.trade_context['size'] > 0:
                    entry_price = self.trade_context['entry_price']
                    size_closed = self.trade_context['size']
                    side_closed = self.trade_context['side']
                    entry_reason = self.trade_context['entry_reason']
                else:
                    entry_price = pos_data['entry_price']
                    size_closed = pos_data['size']
                    side_closed = pos_data['side']
                    entry_reason = "Unknown (Bot Restarted)" 

                if side_closed == 'LONG':
                    raw_pnl = (exit_price - entry_price) * size_closed
                else: 
                    raw_pnl = (entry_price - exit_price) * size_closed
                
                fee_cost = (entry_price * size_closed * 0.00035) + (exit_price * size_closed * 0.00035)
                net_pnl = raw_pnl - fee_cost
                
                if self.last_entry_time:
                    duration = datetime.now() - self.last_entry_time
                else:
                    duration = timedelta(0)
                
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

                self.last_entry_time = None
                self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE'}
                self.risk_manager.update_position_tracking(0, "CLOSE")
                self._save_bot_state()

        else:
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
                final_reasoning = reasoning + ai_forecast_info
                self.trade_context = {
                    'entry_price': current_price,
                    'entry_reason': final_reasoning,
                    'size': size,
                    'side': side,
                    'sl_percent': sl_percent  
                }
                self.last_entry_time = datetime.now()
                self.risk_manager.update_position_tracking(order_value, "ADD")
                self._save_bot_state()
            else:
                print("❌ 取引失敗")

        # === 9. Google Sheetsログ記録 (Executions/AI/Equity) ===
        # ★ここでATRを計算
        atr_pct = (analysis.get('indicators', {}).get('atr', 0) / current_price * 100) if current_price > 0 else 0

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
                'reasoning': reasoning + ai_forecast_info,
                'status': 'EXECUTED' if trade_success else 'FAILED'
            },
            signal_data={
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': action,
                'confidence': confidence,
                'ml_probabilities': decision.get('ml_probabilities', {}),
                'price': current_price,
                'volatility': atr_pct, 
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
                self._save_bot_state()
                
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
            fast_interval = 1 
            ai_loop_count = 0

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
                    self.risk_manager.current_capital = float(account_state.get('crossMarginSummary', {}).get('accountValue', 0)) or float(account_state.get('marginSummary', {}).get('accountValue', 0))
                    
                    # ポジション情報の取得
                    pos_data = self._get_position_summary(account_state)
                    
                    # 建値ストップ (Breakeven Stop) の実装
                    # 0-15分の間でも、利益が出たら逃げ道を確保する
                    if pos_data['found']:
                        entry_px = pos_data['entry_price']
                        # 利益率(%)の計算
                        if pos_data['side'] == 'LONG':
                            pnl_pct = (current_price - entry_px) / entry_px * 100
                        else:
                            pnl_pct = (entry_px - current_price) / entry_px * 100
                        
                        # A. 利益が +0.2% を超えたら「建値ガード」を有効化
                        if pnl_pct > 0.2 and not self.trade_context.get('breakeven_active'):
                            self.trade_context['breakeven_active'] = True
                            print(f"🔒 建値ガード発動: 利益確保モードへ移行 (現在:{pnl_pct:.2f}%)")
                            self._save_bot_state()

                        # B. ガード有効中に +0.10% (手数料分) を割りそうになったら決済
                        if self.trade_context.get('breakeven_active', False) and pnl_pct < 0.10:
                            print(f"🛡️ 建値撤退実行: 勝ち逃げ ({pnl_pct:.2f}%)")
                            self.trader.close_position(self.symbol)
                            
                            # 状態リセット
                            self.last_entry_time = None
                            self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE'}
                            self._save_bot_state()
                            time.sleep(fast_interval)
                            continue

                    else:
                        # ポジションがない場合はフラグをリセット
                        if self.trade_context.get('breakeven_active'):
                            self.trade_context['breakeven_active'] = False

                    # 日次リセットや緊急停止チェック
                    self.check_daily_exit(account_state)
                    if pos_data['found']:
                        self._check_emergency_exit(pos_data, current_price)

                # --- AI判断フェーズ (interval秒ごと) ---
                if (current_time - last_ai_check_time >= interval) or (last_ai_check_time == 0):
                    
                    # モデルのホットリロード
                    ai_loop_count += 1
                    if self.ml_predictor and (ai_loop_count % 10 == 0):
                         try: 
                             print("🔄 モデルを再読み込み中...") 
                             self.ml_predictor.load_models()
                         except: pass

                    print(f"\n{'='*70}")
                    print(f"📊 {self.symbol} Price: ${current_price:.2f}")
                    
                    # 1. 市場分析データの取得
                    analysis = self.market_data.get_comprehensive_analysis()
                    
                    # 2. 板情報 (Structure) の取得
                    structure = self.market_data.get_market_structure_features()
                    fast_imbalance = self.ws_monitor.get_latest_imbalance()

                    # 3. OIの取得と変化率計算
                    current_oi = self.ws_monitor.get_latest_oi()
                    if current_oi == 0:
                        current_oi = self.market_data.get_open_interest()

                    oi_delta_pct = 0.0
                    if self.last_oi > 0:
                        oi_delta_pct = ((current_oi - self.last_oi) / self.last_oi) * 100
                    
                    if current_oi > 0:
                        self.last_oi = current_oi
                    
                    if analysis:
                        # --- ATRベースの高感度ボラティリティ判定 ---
                        tf_data = analysis['timeframes'].get(MAIN_TIMEFRAME, {})
                        atr_val = tf_data.get('atr', 0)
                        
                        if current_price > 0:
                            atr_pct = (atr_val / current_price) * 100
                        else:
                            atr_pct = 0.0

                        volatility = analysis.get('volatility', 0)

                        print(f"   ATR(15m): {atr_pct:.3f}% (${atr_val:.2f}) | StdVol(15m): {volatility:.2f}%")
                        print(f"   Imb: {fast_imbalance:.2f} | OI: {current_oi:.0f} | OI Δ: {oi_delta_pct:+.4f}%")

                        # 3. 閾値判定
                        MIN_ATR_LIMIT = 0.3 
                        
                        if atr_pct < MIN_ATR_LIMIT:
                            status_msg = f"💤 低ボラティリティ待機 (ATR: {atr_pct:.3f}% < {MIN_ATR_LIMIT}%)"
                            print(status_msg)

                            signal_log = {
                                'timestamp': datetime.now(),
                                'symbol': self.symbol,
                                'action': 'WAIT',
                                'confidence': 0,
                                'ml_probabilities': {'up': 0.0, 'down': 0.0},
                                'price': current_price,
                                'volatility': atr_pct,
                                'rsi': analysis.get('indicators', {}).get('rsi', 0),
                                'market_regime': 'LOW_VOLATILITY', 
                                'model_used': '-',
                                'price_diff': '-',
                                'prediction_result': status_msg 
                            }
                            self.log_to_sheets(signal_data=signal_log)

                            last_ai_check_time = current_time 
                            time.sleep(fast_interval)
                            continue
                        
                        # 3. ML判断を実行
                        structure['oi_delta_pct'] = oi_delta_pct
                        decision = self.get_ml_decision(analysis, account_state, structure)
                        
                        if decision:
                            action     = decision.get('action', 'HOLD')
                            confidence = decision.get('confidence', 0)
                            up_prob    = decision['ml_probabilities']['up']
                            down_prob  = decision['ml_probabilities']['down']
                            
                            # === 予測判定に「理由」を表示する ===
                            # デフォルトはAIが算出したreasoning (Why No Trade / Why Hold)
                            prediction_result = decision.get('reasoning', '-')
                            
                            # フィルタリング理由があればそれを優先
                            if decision.get('filter_reason'):
                                prediction_result = f"⛔ {decision['filter_reason']}"

                            # AIの予想変動率があれば追記する
                            # 例: "Hold: LONG継続 (Down:0.40) (予:+0.15%)"
                            pred_change = float(decision.get('predicted_change', 0.0))
                            if pred_change != 0:
                                prediction_result += f" (予:{pred_change:+.2f}%)"

                            # decisionに結果を格納
                            decision['prediction_result'] = prediction_result
                            
                            # 4. AI思考ログの作成
                            signal_log = {
                                'timestamp': datetime.now(),
                                'symbol': self.symbol,
                                'action': action,
                                'confidence': confidence,
                                'ml_probabilities': decision.get('ml_probabilities'),
                                'price': current_price,
                                'volatility': atr_pct,
                                'rsi': analysis.get('indicators', {}).get('rsi', 0),
                                'market_regime': decision.get('market_regime'),
                                'model_used': decision.get('reasoning', '').split('|')[-1].strip(),
                                'price_diff': '-',
                                'prediction_result': prediction_result 
                            }
                            
                            # 5. 取引実行 または ログ記録のみ
                            if action == "CLOSE":
                                self.execute_trade(decision, current_price, account_state, analysis)
                            
                            elif action in ['BUY', 'SELL']:
                                self.execute_trade(decision, current_price, account_state, analysis)

                            else:
                                self.log_to_sheets(signal_data=signal_log)

                        else:
                            print("⚠️ ML判断不能")

                    last_ai_check_time = current_time
                    print(f"⏳ 待機中...")

                time.sleep(fast_interval)
                
        except KeyboardInterrupt:
            print("\n⏸️ 停止")
            if self.sheets_logger:
                self.sheets_logger.force_flush()

            self.online_learner.stop_background_learning()
            self.running = False

    def _check_emergency_exit(self, pos_data, current_price):
        """
        緊急決済ロジック
        """
        entry_px = pos_data['entry_price']
        side = pos_data['side']
        size = pos_data['size']
        
        if side == 'LONG':
            pnl_pct = ((current_price - entry_px) / entry_px * 100)
        else:
            pnl_pct = ((entry_px - current_price) / entry_px * 100)
        
        mem_sl = self.trade_context.get('sl_percent', None)
        if mem_sl is not None:
            current_sl_threshold = -abs(float(mem_sl))
            sl_source = "Dynamic(AI)"
        else:
            current_sl_threshold = EMERGENCY_SL_PCT
            sl_source = "Emergency(Global)"

        if pnl_pct <= current_sl_threshold:
            print(f"🚨 {sl_source} 損切り実行: {pnl_pct:.2f}% (閾値: {current_sl_threshold}%)")
            self.trader.close_position(self.symbol)
            pnl_amount = (current_price - entry_px) * size if side == 'LONG' else (entry_px - current_price) * size
            self.log_to_sheets(trade_data={
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': 'CLOSE',
                'side': side,
                'size': size,
                'price': current_price,
                'order_value': size * current_price,
                'fee': 0, 
                'realized_pnl': pnl_amount,
                'unrealized_pnl': 0,
                'confidence': 0,
                'signal_strength': 0,
                'leverage': 0,
                'balance': 0,
                'reasoning': f'{sl_source} Stop Loss ({pnl_pct:.2f}%)',
                'status': 'EXECUTED'
            })
            self.risk_manager.update_position_tracking(0, "CLOSE")
            self.last_entry_time = None
            self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE', 'sl_percent': None}

        elif pnl_pct >= SECURE_PROFIT_TP_PCT:
            print(f"🎉 緊急利確実行: {pnl_pct:.2f}% (閾値: {SECURE_PROFIT_TP_PCT}%)")
            self.trader.close_position(self.symbol)
            pnl_amount = (current_price - entry_px) * size if side == 'LONG' else (entry_px - current_price) * size
            self.log_to_sheets(trade_data={
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': 'CLOSE',
                'side': side,
                'size': size,
                'price': current_price,
                'order_value': size * current_price,
                'fee': 0,
                'realized_pnl': pnl_amount,
                'unrealized_pnl': 0,
                'confidence': 0,
                'signal_strength': 0,
                'leverage': 0,
                'balance': 0,
                'reasoning': f'Emergency Take Profit ({pnl_pct:.2f}%)',
                'status': 'EXECUTED'
            })
            self.risk_manager.update_position_tracking(0, "CLOSE")
            self.last_entry_time = None
            self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE', 'sl_percent': None}
            self._save_bot_state()

    def _get_position_summary(self, account_state: dict) -> dict:
        """
        ポジション情報を一括取得
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


if __name__ == "__main__":
    main()