# main.py
# Hyperliquid 自動トレーディングボット (.env完全対応版)

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

# 時間軸設定
MAIN_TIMEFRAME = os.getenv('MAIN_TIMEFRAME', '15m')  # デイトレードの主軸
SUB_TIMEFRAME = '1h'                                 # トレンド確認・大きな波用

# 緊急損切り・利確設定を.envから取得 (変数の定義漏れを修正)
EMERGENCY_SL_PCT = float(os.getenv('EMERGENCY_STOP_LOSS', '-3.0'))
SECURE_PROFIT_TP_PCT = float(os.getenv('SECURE_TAKE_PROFIT', '6.0'))

class TradingBot:
    """
    Hyperliquid 自動トレーディングボット (デイトレード特化版)
    LightGBM + LSTM によるアンサンブル予測 + マルチタイムフレーム分析
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
            'tp_percent': None,
            'timeframe': '15m'  # エントリー根拠となった時間軸
        }

        # 状態保存ファイルのパス
        self.state_file = "bot_state.json"
        self.last_prediction_state = {
            '15m': None, 
            '1h': None
        }

        # 機械学習予測器
        print("🔄 モデル読み込み中...")
        self.ml_15m = MLPredictor(symbol=symbol, timeframe='15m')
        self.ml_1h  = MLPredictor(symbol=symbol, timeframe='1h')
        
        # 学習機能
        self.online_learner = OnlineLearner(symbol=symbol, timeframe='15m', retrain_interval_hours=4)
        
        print(f"🤖 機械学習予測システム: 有効 (15m & 1h)")
        
        # Google Sheetsロガー初期化
        self.sheets_logger = None
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

        # 起動時に前回の状態を復元する
        self._load_bot_state()

        print("\n" + "="*70)
        print(f"🚀 Hyperliquid {self.bot_name} Bot (Multi-Timeframe Logic)")
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
            account_state = self.trader.get_user_state()
            pos_data = self._get_position_summary(account_state)
            
            if not pos_data['found']:
                # ポジションがないのにデータが残っていたらクリア
                if self.last_entry_time is not None:
                    print("⚠️ ポジション不整合を検知: 状態をリセットします")
                    self.last_entry_time = None
                    self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE', 'timeframe': '15m'}
                    self._save_bot_state()
            
        except Exception as e:
            print(f"⚠️ 状態復元エラー: {e}")

    # -----------------------------------------------------------
    # ヘルパー: 前回の答え合わせ
    # -----------------------------------------------------------
    def _evaluate_last_prediction(self, current_price: float, timeframe: str) -> str:
        """
        前回の予測が正しかったか答え合わせをする
        (手数料 0.1% を考慮して、Holdが正解だったかを判定)
        """
        last_state = self.last_prediction_state.get(timeframe)
        if not last_state:
            return "-"

        last_price = last_state['price']
        last_action = last_state['action']
        up_prob = last_state['up_prob']
        down_prob = last_state['down_prob']
        
        # 変動率 (%)
        pct_change = (current_price - last_price) / last_price * 100
        
        # 手数料目安 (往復 + スリッページ)
        FEE_COST = 0.1
        
        result_text = "-"

        # ケースA: 前回 HOLD だった場合
        if last_action == 'HOLD':
            # === AIが「上昇」寄りだった場合 ===
            if up_prob > down_prob:
                # 実際の利益(手数料引き後)
                net_profit = pct_change - FEE_COST
                
                if net_profit > 0:
                    # 手数料を引いてもプラス -> エントリーすべきだった
                    result_text = f"🔼 機会損失 (Long利幅 +{net_profit:.2f}% ※手数料引)"
                elif pct_change < -0.1:
                    # 明らかに下がった
                    result_text = f"❌ 予測失敗 (上昇予想も下落 {pct_change:.2f}%)"
                else:
                    # 上がったが手数料負け、または微減 -> Holdで正解
                    if pct_change >= 0:
                        result_text = f"✅ 正解Hold (手数料負け回避 +{pct_change:.2f}%)"
                    else:
                        result_text = f"✅ 正解Hold (微減回避 {pct_change:.2f}%)"

            # === AIが「下落」寄りだった場合 ===
            else:
                # Shortの場合、価格下落が利益 (手数料引き後)
                net_profit = (-pct_change) - FEE_COST
                
                if net_profit > 0:
                    # 手数料を引いてもプラス
                    result_text = f"❌ 機会損失 (Short利幅 +{net_profit:.2f}% ※手数料引)"
                elif pct_change > 0.1:
                    # 明らかに上がった
                    result_text = f"❌ 予測失敗 (下落予想も上昇 +{pct_change:.2f}%)"
                else:
                    # 下がったが手数料負け、または微増 -> Holdで正解
                    if pct_change <= 0:
                        result_text = f"✅ 正解Hold (手数料負け回避 {pct_change:.2f}%)"
                    else:
                        result_text = f"✅ 正解Hold (微増回避 +{pct_change:.2f}%)"

        # ケースB: 前回 BUY だった場合
        elif last_action == 'BUY':
            # 手数料(0.1%)を引いて利益が出ているか
            real_pnl = pct_change - 0.1
            if real_pnl > 0: result_text = f"✅ 勝利 (+{real_pnl:.2f}%)"
            else: result_text = f"❌ 敗北 ({real_pnl:.2f}%)"

        # ケースC: 前回 SELL だった場合
        elif last_action == 'SELL':
            real_pnl = -pct_change - 0.1
            if real_pnl > 0: result_text = f"✅ 勝利 (+{real_pnl:.2f}%)"
            else: result_text = f"❌ 敗北 ({real_pnl:.2f}%)"

        return result_text

    # -----------------------------------------------------------
    # 1. 実行ループ: 「常時監視」と「各時間軸のAI判定」を管理
    # -----------------------------------------------------------
    def run_trading_loop(self, interval=10):
        self.running = True
        self.online_learner.start_background_learning()
        
        print(f"\n🚀 自動トレーディング開始")
        print(f"   監視間隔: {interval}秒 (損切りチェック)")
        print(f"   監視時間軸: {MAIN_TIMEFRAME}, {SUB_TIMEFRAME}")
        
        last_candle_15m = None
        last_candle_1h = None
        
        try:
            while self.running:
                # === A. 常時実行フェーズ (価格監視・損切り) ===
                current_time = datetime.now()
                current_price = self.trader.get_current_price(self.symbol)
                account_state = self.trader.get_user_state()
                
                if not current_price:
                    time.sleep(interval)
                    continue

                # ポジション情報の取得
                pos_data = {'found': False}
                if account_state:
                    account_value = float(account_state.get('crossMarginSummary', {}).get('accountValue', 0)) or float(account_state.get('marginSummary', {}).get('accountValue', 0))
                    
                    pos_data = self._get_position_summary(account_state)
                    
                    # RiskManagerへ同期
                    self.risk_manager.sync_account_state(account_value, pos_data['position_value'])
                    
                    if pos_data['found']:
                        self._check_emergency_exit(pos_data, current_price)
                        self.check_daily_exit(account_state)
                        
                        elapsed = (datetime.now() - self.last_entry_time).total_seconds() / 60 if self.last_entry_time else 0
                        current_tf = self.trade_context.get('timeframe', '15m')
                        time_limit = 60 if current_tf == '15m' else 240
                        
                        if elapsed > time_limit: 
                            print(f"⏰ {time_limit}分経過 ({current_tf}): タイムリミット決済")
                            self.trader.close_position(self.symbol)
                            self.last_entry_time = None
                            time.sleep(interval)
                            continue

                # === B. 時間軸ごとの判定フェーズ ===
                
                # 15分足の確定チェック
                min_15 = (current_time.minute // 15) * 15
                curr_15m = current_time.replace(minute=min_15, second=0, microsecond=0)
                is_new_15m = (last_candle_15m is not None) and (last_candle_15m != curr_15m)

                # 1時間足の確定チェック
                curr_1h = current_time.replace(minute=0, second=0, microsecond=0)
                is_new_1h = (last_candle_1h is not None) and (last_candle_1h != curr_1h)
                
                # 初回起動時は基準時刻セットのみ
                if last_candle_15m is None:
                    last_candle_15m = curr_15m
                    last_candle_1h = curr_1h
                    
                    # 表示用に来るべき時間を計算 (+15分, +1時間)
                    next_target_15m = curr_15m + timedelta(minutes=15)
                    next_target_1h = curr_1h + timedelta(hours=1)
                    
                    print(f"⏳ 次の足確定を待機中... (Target >> 15m: {next_target_15m.strftime('%H:%M')}, 1h: {next_target_1h.strftime('%H:%M')})")
                    time.sleep(interval)
                    continue

                # AI判断結果を格納する変数
                decision_15m = None
                decision_1h = None

                # --- 15分足のAI予測 ---
                if is_new_15m:
                    print(f"\n⏰ 15分足確定 ({curr_15m.strftime('%H:%M')})")
                    last_candle_15m = curr_15m
                    
                    try:
                        analysis_15m = self.market_data.get_comprehensive_analysis(interval='15m')
                    except:
                        analysis_15m = self.market_data.get_comprehensive_analysis()
                    
                    if analysis_15m: analysis_15m['price'] = current_price
                    
                    eval_result = self._evaluate_last_prediction(current_price, '15m')

                    # 予測実行
                    decision_15m = self.get_ml_decision(self.ml_15m, analysis_15m, account_state, '15m')
                    
                    # 今回の予測内容を保存
                    self.last_prediction_state['15m'] = {
                        'price': current_price,
                        'action': decision_15m['action'],
                        'up_prob': decision_15m['ml_probabilities'].get('up', 0),
                        'down_prob': decision_15m['ml_probabilities'].get('down', 0)
                    }
                    
                    # ログ記録
                    self.log_to_sheets(signal_data={
                        'timestamp': datetime.now(),
                        'timeframe': '15m',
                        'symbol': self.symbol,
                        'price': current_price,
                        'eval_result': eval_result,
                        **decision_15m 
                    })

                    time.sleep(3)

                # --- 1時間足のAI予測 ---
                if is_new_1h:
                    print(f"\n🔔 1時間足確定 ({curr_1h.strftime('%H:%M')})")
                    last_candle_1h = curr_1h
                    
                    try:
                        analysis_1h = self.market_data.get_comprehensive_analysis(interval='1h')
                    except:
                        analysis_1h = self.market_data.get_comprehensive_analysis()
                    
                    if analysis_1h: analysis_1h['price'] = current_price

                    # 前回の答え合わせ
                    eval_result = self._evaluate_last_prediction(current_price, '1h')
                    
                    # 予測実行
                    decision_1h = self.get_ml_decision(self.ml_1h, analysis_1h, account_state, '1h')

                    # 今回の予測内容を保存
                    self.last_prediction_state['1h'] = {
                        'price': current_price,
                        'action': decision_1h['action'],
                        'up_prob': decision_1h['ml_probabilities'].get('up', 0),
                        'down_prob': decision_1h['ml_probabilities'].get('down', 0)
                    }
                    
                    # ログ記録
                    self.log_to_sheets(signal_data={
                        'timestamp': datetime.now(),
                        'timeframe': '1h',
                        'symbol': self.symbol,
                        'price': current_price,
                        'eval_result': eval_result,
                        **decision_1h
                    })

                # === C. エントリー・決済の統合判断 ===
                if not pos_data['found']:
                    target_decision = None
                    target_tf = '15m'
                    
                    if decision_1h and decision_1h['action'] in ['BUY', 'SELL']:
                        print("✨ 1時間足でチャンス発生！ (優先採用)")
                        target_decision = decision_1h
                        target_tf = '1h'
                    
                    elif decision_15m and decision_15m['action'] in ['BUY', 'SELL']:
                        print("✨ 15分足でチャンス発生！")
                        target_decision = decision_15m
                        target_tf = '15m'
                    
                    if target_decision:
                        self.execute_trade(target_decision, current_price, account_state, {}, timeframe=target_tf)

                else:
                    entry_tf = self.trade_context.get('timeframe', '15m')
                    
                    check_decision = None
                    if entry_tf == '15m' and decision_15m:
                        check_decision = decision_15m
                    elif entry_tf == '1h' and decision_1h:
                        check_decision = decision_1h
                    
                    if check_decision:
                        action = check_decision.get('action')
                        print(f"🧐 継続審査 ({entry_tf}): {action} - {check_decision.get('reasoning')}")
                        
                        if action in ['CLOSE', 'BUY', 'SELL']: 
                            if action == 'CLOSE' or (action != self.trade_context['side']):
                                print(f"🛑 {entry_tf}足による決済/ドテン実行")
                                self.execute_trade(check_decision, current_price, account_state, {}, timeframe=entry_tf)

                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n⏸️ 停止")
            self.online_learner.stop_background_learning()
            self.running = False

    # -----------------------------------------------------------
    # 2. AI判定ロジック: .envから値を取得するように修正済み
    # -----------------------------------------------------------
    def get_ml_decision(self, predictor, market_analysis: dict, account_state: dict, timeframe: str) -> dict:
        """
        AIによる売買判断ロジック (15m / 1h 共通)
        """
        indicators = market_analysis.get('indicators', {})
        rsi = indicators.get('rsi', market_analysis.get('rsi', 50))
        volatility = market_analysis.get('volatility', 0)
        
        try:
            # === データ準備 ===
            df = self.market_data.get_ohlcv(timeframe, limit=200)
            structure = self.market_data.get_market_structure_features()
            
            # 指定されたpredictorで予測
            ml_result = predictor.predict(df, extra_features=structure)
            
            if ml_result.get('model_used') == 'NONE':
                return {
                    'action': 'HOLD', 
                    'confidence': 0, 
                    'reasoning': 'Wait: 未学習 (モデルを作成してください)', 
                    'ml_probabilities': {},
                    'rsi': rsi,
                    'volatility': volatility
                }

            up_prob = ml_result['up_prob']
            down_prob = ml_result['down_prob']
            predicted_change = ml_result.get('predicted_change', 0.0)
            
            current_price = market_analysis.get('price', 0)
            if current_price == 0:
                return {
                    'action': 'HOLD', 'confidence': 0, 'reasoning': 'Wait: 価格取得エラー', 
                    'ml_probabilities': {}, 'rsi': rsi, 'volatility': volatility
                }

            existing_side = None
            if account_state and 'assetPositions' in account_state:
                for pos in account_state['assetPositions']:
                    p = pos.get('position', {})
                    if p.get('coin') == self.symbol and float(p.get('szi', 0)) != 0:
                        existing_side = 'LONG' if float(p.get('szi', 0)) > 0 else 'SHORT'
                        break

            # 期待値(EV)計算
            ev_score_up = up_prob * abs(predicted_change)
            ev_score_down = down_prob * abs(predicted_change)
            
            # ★修正: 閾値を.envから取得
            EV_THRESHOLD = float(os.getenv('ENTRY_EV_THRESHOLD', 0.12))
            PROB_THRESHOLD = float(os.getenv('ENTRY_PROB_THRESHOLD', 0.38))
            HIGH_PROB_THRESHOLD = float(os.getenv('ENTRY_HIGH_PROB_THRESHOLD', 0.45))

            # 今のトレンド判定
            current_trend = "NONE"
            if up_prob > down_prob and up_prob > PROB_THRESHOLD:
                # 期待値 > EV_THRESHOLD または 確率 > HIGH_PROB_THRESHOLD
                if ev_score_up > EV_THRESHOLD or up_prob > HIGH_PROB_THRESHOLD: 
                    current_trend = "BUY"
            elif down_prob > up_prob and down_prob > PROB_THRESHOLD:
                if ev_score_down > EV_THRESHOLD or down_prob > HIGH_PROB_THRESHOLD:
                    current_trend = "SELL"

            action = 'HOLD'
            side = 'NONE'
            reasoning = ""
            confidence = ml_result['confidence']

            # A: 継続審査
            if existing_side:
                if existing_side == 'LONG' and current_trend == 'SELL':
                    action = 'SELL'
                    side = 'SHORT'
                    reasoning = f"Switch: 上昇終了判定 (Down:{down_prob:.2f})"
                elif existing_side == 'SHORT' and current_trend == 'BUY':
                    action = 'BUY'
                    side = 'LONG'
                    reasoning = f"Switch: 下落終了判定 (Up:{up_prob:.2f})"
                elif current_trend == 'NONE':
                    action = 'CLOSE'
                    reasoning = f"CLOSE: トレンド消滅 (Up:{up_prob:.2f}/Down:{down_prob:.2f})"
                else:
                    action = 'HOLD'
                    side = existing_side
                    reasoning = f"Keep: トレンド継続中 ({current_trend})"

            # B: 新規エントリー
            else:
                if current_trend == "BUY":
                    action = "BUY"
                    side = "LONG"
                    reasoning = f"Entry BUY: EV({ev_score_up:.3f})"
                elif current_trend == "SELL":
                    action = "SELL"
                    side = "SHORT"
                    reasoning = f"Entry SELL: EV({ev_score_down:.3f})"
                else:
                    action = "HOLD"
                    # Wait理由 (.envの値を使って計算)
                    up_pct = up_prob * 100
                    down_pct = down_prob * 100
                    thresh_pct = PROB_THRESHOLD * 100
                    
                    if up_prob > down_prob:
                        if up_prob <= PROB_THRESHOLD:
                            reasoning = f"Wait: 確率不足 (Up:{up_pct:.0f}% < 基準{thresh_pct:.0f}%)"
                        else:
                            reasoning = f"Wait: EV不足 (Up EV:{ev_score_up:.3f} < 基準{EV_THRESHOLD})"
                    else:
                        if down_prob <= PROB_THRESHOLD:
                            reasoning = f"Wait: 確率不足 (Down:{down_pct:.0f}% < 基準{thresh_pct:.0f}%)"
                        else:
                            reasoning = f"Wait: EV不足 (Down EV:{ev_score_down:.3f} < 基準{EV_THRESHOLD})"

            # リスクパラメータ
            volatility = market_analysis.get('volatility', 2.0)
            if volatility > 3.0: sl_pct, tp_pct = 2.0, 3.5 
            else: sl_pct, tp_pct = 1.0, 2.0
            
            win_prob = up_prob if action == 'BUY' else down_prob

            return {
                'action': action,
                'side': side,
                'confidence': confidence,
                'expected_value_r': (win_prob * tp_pct) - ((1 - win_prob) * sl_pct),
                'risk_reward_ratio': tp_pct / sl_pct,
                'stop_loss_percent': sl_pct,
                'take_profit_percent': tp_pct,
                'reasoning': f"{reasoning} | {ml_result['model_used']}",
                'ml_probabilities': {'up': up_prob, 'down': down_prob},
                'predicted_change': predicted_change,
                'market_regime': 'NORMAL',
                'volatility': volatility,
                'rsi': rsi, 
                'prediction_result': reasoning
            }
            
        except Exception as e:
            print(f"⚠️ ML判断エラー: {e}")
            return {
                'action': 'HOLD', 
                'confidence': 0, 
                'reasoning': f'Error: {str(e)}', 
                'ml_probabilities': {},
                'rsi': rsi,
                'volatility': volatility
            }

    # -----------------------------------------------------------
    # 3. ログ・実行・管理メソッド (Timeframe対応版)
    # -----------------------------------------------------------
    def log_to_sheets(self, trade_data: dict = None, signal_data: dict = None, snapshot_data: dict = None):
        """Google Sheetsにログを記録"""
        if not self.enable_sheets_logging or not self.sheets_logger: return
        try:
            if trade_data: self.sheets_logger.log_execution(trade_data)
            if signal_data:
                tf = signal_data.get('timeframe', '15m')
                probs = signal_data.get('ml_probabilities', {})
                analysis_payload = {
                    'timestamp': signal_data.get('timestamp'),
                    'timeframe': tf,
                    'price': signal_data.get('price'),
                    'action': signal_data.get('action', 'HOLD'),
                    'confidence': signal_data.get('confidence', 0),
                    'up_prob': probs.get('up', 0),
                    'down_prob': probs.get('down', 0),
                    'market_regime': signal_data.get('market_regime', 'NORMAL'),
                    'model_used': signal_data.get('model_used', 'ENSEMBLE'),
                    'rsi': signal_data.get('rsi', 0),
                    'volatility': signal_data.get('volatility', 0),
                    'eval_result': signal_data.get('eval_result', '-'),
                    'prediction_result': signal_data.get('prediction_result', '-')
                }
                self.sheets_logger.log_ai_analysis(analysis_payload)
            if snapshot_data:
                self.sheets_logger.log_equity(snapshot_data)
        except Exception as e:
            print(f"⚠️ Google Sheetsログ記録エラー: {e}")

    def _log_cancel_reason(self, decision, current_price, analysis, reason_text, timeframe='15m'):
        """トレード拒否理由を記録"""
        atr_pct = (analysis.get('indicators', {}).get('atr', 0) / current_price * 100) if current_price > 0 else 0
        eval_result = self._evaluate_last_prediction(current_price, timeframe)

        signal_log = {
            'timestamp': datetime.now(),
            'timeframe': timeframe,
            'symbol': self.symbol,
            'action': 'WAIT',
            'confidence': decision.get('confidence', 0),
            'ml_probabilities': decision.get('ml_probabilities', {}),
            'price': current_price,
            'volatility': atr_pct,
            'rsi': analysis.get('indicators', {}).get('rsi', 0),
            'market_regime': 'NORMAL',
            'model_used': decision.get('reasoning', '').split('|')[-1].strip(),
            'eval_result': eval_result,
            'prediction_result': f"⛔ {reason_text}"
        }
        self.log_to_sheets(signal_data=signal_log)

    def execute_trade(self, decision: dict, current_price: float, account_state: dict, analysis: dict, timeframe: str = '15m'):
        """実際の取引を実行してGoogle Sheetsに記録"""
        action = decision.get('action')
        ev = float(decision.get('expected_value_r', 0))
        rr_ratio = float(decision.get('risk_reward_ratio', 0))
        
        # 簡易フィルタ
        ESTIMATED_COST_PCT = 0.1
        net_ev = ev - ESTIMATED_COST_PCT
        if action in ['BUY', 'SELL']:
            if net_ev <= 0.05: 
                reason = f"EV不足(Net:{net_ev:.2f}%)"
                print(f"🛑 取引拒否: {reason}")
                self._log_cancel_reason(decision, current_price, analysis, reason, timeframe)
                return
            if rr_ratio < 0.8:
                reason = f"RR不足({rr_ratio:.2f})"
                print(f"🛑 取引拒否: {reason}")
                self._log_cancel_reason(decision, current_price, analysis, reason, timeframe)
                return
        
        # 資金情報
        cross_margin = account_state.get('crossMarginSummary', {}) if account_state else {}
        margin_summary = account_state.get('marginSummary', {}) if account_state else {}
        account_value = float(cross_margin.get('accountValue', 0)) or float(margin_summary.get('accountValue', 0))
        available_balance = float(cross_margin.get('totalRawUsd', 0)) or float(margin_summary.get('totalRawUsd', 0))
        
        self.risk_manager.current_capital = account_value
        pos_data = self._get_position_summary(account_state)
        existing_position_value = pos_data['position_value']
        unrealized_pnl = pos_data['unrealized_pnl']
        
        # 1. 日次損失制限
        if not self.risk_manager.check_daily_loss_limit():
            self._log_cancel_reason(decision, current_price, analysis, "日次損失限度到達", timeframe)
            return
        
        confidence = float(decision.get('confidence', 0))
        
        # 2. 追加可否 (CLOSE以外)
        if action != 'CLOSE' and existing_position_value > 0:
            if not self.risk_manager.should_add_position(confidence, existing_position_value):
                self._log_cancel_reason(decision, current_price, analysis, "既存Posあり追加不可", timeframe)
                return
        
        sl_percent = float(decision.get('stop_loss_percent', 2.0))
        tp_percent = float(decision.get('take_profit_percent', 3.0))
        side = decision.get('side')
        
        size = 0.0
        risk_level = "CLOSE"
        reasoning = decision.get('reasoning')
        order_value = 0.0
        ai_forecast_info = ""

        if action != 'CLOSE':
            print(f"\n{'='*70}\n🔍 サイズ計算 ({timeframe})\n{'='*70}")
            position_result = self.risk_manager.calculate_position_size_by_confidence(
                capital=account_value, entry_price=current_price, confidence=confidence,
                existing_position_value=existing_position_value, stop_loss_percent=sl_percent,
                max_available_cash=available_balance
            )
            size        = position_result['size']
            risk_level  = position_result['risk_level']
            reasoning   = position_result['reasoning']
            order_value = position_result['position_value']
            
            predicted_change = float(decision.get('predicted_change', 0.0))
            if side == 'LONG': target_change_pct = abs(predicted_change) if predicted_change != 0 else 0.5 
            else: target_change_pct = -abs(predicted_change) if predicted_change != 0 else -0.5
            
            expected_price = current_price * (1 + target_change_pct / 100)
            expected_profit = abs(expected_price - current_price) * size
            
            print(f"   サイズ: {size:.4f} ETH (${order_value:.2f})")
            print(f"   予想: {target_change_pct:+.2f}% (益 ${expected_profit:.2f})")
            
            if size == 0:
                self._log_cancel_reason(decision, current_price, analysis, "サイズ計算結果0", timeframe)
                return
            
            ai_forecast_info = f" | 🔮予:{target_change_pct:+.2f}% 💰${expected_profit:.2f}"

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
                    entry_reason = "Unknown" 
                
                if side_closed == 'LONG': raw_pnl = (exit_price - entry_price) * size_closed
                else: raw_pnl = (entry_price - exit_price) * size_closed
                fee_cost = (entry_price * size_closed * 0.00035) + (exit_price * size_closed * 0.00035)
                net_pnl = raw_pnl - fee_cost
                
                duration = datetime.now() - self.last_entry_time if self.last_entry_time else timedelta(0)
                
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
                self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE', 'timeframe': '15m'}
                self.risk_manager.update_position_tracking(0, "CLOSE")
                self._save_bot_state()

        else:
            print(f"🛡️ 注文送信中 ({side})...")
            is_buy = (side == 'LONG')
            result = self.trader.place_limit_order(
                symbol=self.symbol, is_buy=is_buy, size=size,
                time_in_force="Ioc", aggressive=True 
            )
            estimated_fee = order_value * 0.00035
            trade_success = result and result.get('status') == 'ok'
            if trade_success:
                print("✅ 取引成功!")
                self.trade_context = {
                    'entry_price': current_price,
                    'entry_reason': reasoning + ai_forecast_info,
                    'size': size,
                    'side': side,
                    'sl_percent': sl_percent,
                    'timeframe': timeframe
                }
                self.last_entry_time = datetime.now()
                self.risk_manager.update_position_tracking(order_value, "ADD")
                self._save_bot_state()
            else:
                print("❌ 取引失敗")

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
                'balance': available_balance,
                'reasoning': reasoning + ai_forecast_info
            },
            signal_data={
                'timestamp': datetime.now(),
                'timeframe': timeframe,
                'symbol': self.symbol,
                'action': action,
                'confidence': confidence,
                'ml_probabilities': decision.get('ml_probabilities', {}),
                'price': current_price,
                'prediction_result': decision.get('prediction_result', '-')
            },
            snapshot_data={
                'timestamp': datetime.now(),
                'account_value': account_value,
                'available_balance': available_balance,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl_cumulative': 0,
                'position_size': size if trade_success and action != 'CLOSE' else 0,
            }
        )

    def check_daily_exit(self, account_state: dict):
        now = datetime.utcnow()
        if now.hour == 23 and now.minute >= 55:
            pos_data = self._get_position_summary(account_state)
            if pos_data['found']:
                print("\n⏰ 日次強制決済 (UTC 23:55)")
                self.trader.close_position(self.symbol)
                self.last_entry_time = None
                self._save_bot_state()
                self.log_to_sheets(trade_data={
                    'timestamp': datetime.now(),
                    'symbol': self.symbol,
                    'action': 'CLOSE',
                    'reasoning': 'Daily Force Close'
                })
                print("⏳ 翌日まで待機中...")
                time.sleep(300) 

    def _check_emergency_exit(self, pos_data, current_price):
        entry_px = pos_data['entry_price']
        side = pos_data['side']
        size = pos_data['size']
        if side == 'LONG': pnl_pct = ((current_price - entry_px) / entry_px * 100)
        else: pnl_pct = ((entry_px - current_price) / entry_px * 100)
        
        mem_sl = self.trade_context.get('sl_percent', None)
        current_sl_threshold = -abs(float(mem_sl)) if mem_sl is not None else EMERGENCY_SL_PCT
        
        # 保存された時間軸を取得 (なければデフォルト15m)
        tf = self.trade_context.get('timeframe', '15m')

        if pnl_pct <= current_sl_threshold:
            print(f"🚨 損切り実行: {pnl_pct:.2f}%")
            self.trader.close_position(self.symbol)
            pnl_amount = (current_price - entry_px) * size if side == 'LONG' else (entry_px - current_price) * size
            
            self.log_to_sheets(trade_data={
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': 'CLOSE',
                'side': side,
                'size': size,
                'price': current_price,
                'realized_pnl': pnl_amount,
                'reasoning': f'Stop Loss ({pnl_pct:.2f}%)'
            }, signal_data={'timestamp': datetime.now(), 'timeframe': tf, 'action': 'CLOSE', 'prediction_result': 'STOP_LOSS'})
            
            self.risk_manager.update_position_tracking(0, "CLOSE")
            self.last_entry_time = None
            self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE', 'timeframe': '15m'}

        elif pnl_pct >= SECURE_PROFIT_TP_PCT:
            print(f"🎉 緊急利確実行: {pnl_pct:.2f}%")
            self.trader.close_position(self.symbol)
            pnl_amount = (current_price - entry_px) * size if side == 'LONG' else (entry_px - current_price) * size
            
            self.log_to_sheets(trade_data={
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'action': 'CLOSE',
                'side': side,
                'size': size,
                'price': current_price,
                'realized_pnl': pnl_amount,
                'reasoning': f'Take Profit ({pnl_pct:.2f}%)'
            }, signal_data={'timestamp': datetime.now(), 'timeframe': tf, 'action': 'CLOSE', 'prediction_result': 'TAKE_PROFIT'})

            self.risk_manager.update_position_tracking(0, "CLOSE")
            self.last_entry_time = None
            self.trade_context = {'entry_price': 0, 'entry_reason': '', 'size': 0, 'side': 'NONE', 'timeframe': '15m'}
            self._save_bot_state()

    def _get_position_summary(self, account_state: dict) -> dict:
        summary = {'size': 0.0, 'side': 'NONE', 'unrealized_pnl': 0.0, 'entry_price': 0.0, 'position_value': 0.0, 'found': False}
        if not account_state or 'assetPositions' not in account_state: return summary
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
    symbol = os.getenv('TRADING_SYMBOL', 'ETH')
    env_capital = os.getenv('INITIAL_CAPITAL', '1000')
    interval = int(os.getenv('CHECK_INTERVAL', '15'))
    enable_sheets = os.getenv('ENABLE_SHEETS_LOGGING', 'true').lower() == 'true'

    try:
        capital = float(env_capital)
    except ValueError:
        capital = 1000.0
    
    if mode == 'run':
        print(f"\n🚀 Bot起動準備中...")
        bot = TradingBot(symbol=symbol, initial_capital=capital, enable_sheets_logging=enable_sheets)
        bot.run_trading_loop(interval=interval)

if __name__ == "__main__":
    main()