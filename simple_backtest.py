import pandas as pd
import numpy as np
import os
import joblib
import warnings
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model

# Keras / TensorFlow のログを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# === 設定 (main.py チューニング版) ===
CSV_FILE = "backtest_data_ETH_USDT_15m.csv"
INITIAL_CAPITAL = 500.0
FEE_RATE = 0.00035

# エントリー基準を上げて「無駄打ち」を防ぐ
# 0.40だとノイズを拾いすぎるため、上位数%のチャンスに絞る
BASE_THRESHOLD = 0.44

# 撤退ライン
CLOSE_THRESHOLD = 0.55

# フィルター設定
MIN_VOLATILITY_PCT = 0.3

class AccurateBacktester:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.feature_cols = [
            'orderbook_imbalance', 'btc_correlation', 'btc_trend_strength',
            'rsi', 'macd_hist', 'bb_position', 'bb_width',
            'atr', 'volume_ratio', 'price_change_1h',
            'price_change_4h', 'sma_20_50_ratio', 'volatility',
            'hour_sin', 'hour_cos', 'day_of_week'
        ]
        try:
            print("🧠 モデルを読み込み中...")
            self.lgb_model = joblib.load('models/lgb_ETH.pkl')
            self.lstm_model = load_model('models/lstm_ETH.h5', compile=False)
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")

    def load_data(self):
        print(f"📖 データ読み込み中: {self.csv_file}")
        df = pd.read_csv(self.csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def calculate_indicators(self, df):
        print("⚡ テクニカル指標を計算中...")
        df = df.copy()
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal
        
        # BB
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        df['bb_position'] = (close - (sma20 - 2*std20)) / (4*std20)
        df['bb_width'] = (4*std20) / sma20
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - close.shift()).abs()
        tr3 = (df['low'] - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
        df['volatility_pct'] = (df['atr'] / close) * 100
        
        # Others
        sma50 = close.rolling(50).mean()
        df['sma_20_50_ratio'] = (sma20 / sma50 - 1) * 100
        df['volatility'] = close.rolling(20).std() / sma20 * 100
        vol_ma = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / vol_ma.replace(0, 1)
        df['price_change_1h'] = close.pct_change(4) * 100
        df['price_change_4h'] = close.pct_change(16) * 100
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_of_week'] = df['timestamp'].dt.dayofweek / 6.0
        
        # Fill missing
        df['orderbook_imbalance'] = 0.0
        df['btc_correlation'] = 0.0
        df['btc_trend_strength'] = 0.0
        
        return df.fillna(0)

    def predict_probs(self, df):
        print("🧠 AI推論を実行中...")
        X = df[self.feature_cols].values
        lgb_probs = self.lgb_model.predict(X)
        lgb_up = lgb_probs[:, 2]
        lgb_down = lgb_probs[:, 0]
        
        closes = df['close'].values
        returns = np.diff(np.log(closes), prepend=closes[0])
        returns = np.nan_to_num(returns)
        
        window_size = 60
        X_lstm = np.zeros((len(returns), window_size, 1))
        
        n_samples = len(returns) - window_size
        strides = returns.strides + returns.strides
        X_view = np.lib.stride_tricks.as_strided(
            returns, shape=(n_samples, window_size), strides=strides
        )
        X_lstm[window_size:, :, 0] = X_view
        X_lstm = (X_lstm - np.mean(X_lstm)) / (np.std(X_lstm) + 1e-8)

        lstm_probs = self.lstm_model.predict(X_lstm, batch_size=4096, verbose=0)
        
        lstm_up_full = lstm_probs[:, 2]
        lstm_down_full = lstm_probs[:, 0]
        
        df['up_prob'] = lgb_up * 0.6 + lstm_up_full * 0.4
        df['down_prob'] = lgb_down * 0.6 + lstm_down_full * 0.4
        
        return df

    def run(self):
        df = self.load_data()
        
        # カンニング防止: 直近120日を除外
        df = df.iloc[:-11520]
        print(f"📉 テスト期間: {df['timestamp'].iloc[0]} 〜 {df['timestamp'].iloc[-1]}")
        
        df = self.calculate_indicators(df)
        df = self.predict_probs(df)
        
        print("🚀 main.py ロジック（改良版）によるシミュレーション開始...")
        
        balance = INITIAL_CAPITAL
        position = None
        trades = []
        ctx = {}
        
        times = df['timestamp'].values
        prices = df['close'].values
        up_probs = df['up_prob'].values
        down_probs = df['down_prob'].values
        rsis = df['rsi'].values
        vols = df['volatility_pct'].values
        
        for i in range(60, len(df)):
            ts = times[i]
            price = prices[i]
            up_prob = up_probs[i]
            down_prob = down_probs[i]
            rsi = rsis[i]
            vol_pct = vols[i]
            
            # SL/TP 設定 (main.py準拠)
            if vol_pct > 3.0:
                sl_pct, tp_pct = 0.02, 0.035
            else:
                sl_pct, tp_pct = 0.01, 0.02
                
            action = 'HOLD'
            reason = ''
            
            # === ポジション管理 ===
            if position:
                elapsed_min = (ts - position['entry_time']) / np.timedelta64(1, 'm')
                entry_price = position['entry_price']
                side = position['side']
                
                if side == 'LONG': pnl_pct = (price - entry_price) / entry_price
                else: pnl_pct = (entry_price - price) / entry_price
                
                # --- 時間経過判定 (修正版) ---
                
                # 【フェーズ4】45分経過: タイムアップ（デイトレなのでこれは維持）
                if elapsed_min > 45:
                    action = 'CLOSE'
                    reason = 'TimeLimit 45m'
                
                # 【フェーズ3】30分経過: 審査を緩和
                elif elapsed_min >= 30:
                    if not ctx.get('check_30m'):
                        ctx['check_30m'] = True
                        # 厳格審査(+0.02)をやめ、基準値(0.44)を割っていなければOKとする
                        if side == 'LONG' and up_prob < BASE_THRESHOLD:
                            action = 'CLOSE'; reason = '30m Check Failed'
                        elif side == 'SHORT' and down_prob < BASE_THRESHOLD:
                            action = 'CLOSE'; reason = '30m Check Failed'
                
                # 【フェーズ2】15分経過: 「Prob Drop」による撤退を廃止
                # エントリー直後のノイズで狩られるのを防ぐため、何もチェックしない
                
                # --- 常時監視 ---
                if pnl_pct <= -sl_pct: action = 'CLOSE'; reason = 'SL'
                elif pnl_pct >= tp_pct: action = 'CLOSE'; reason = 'TP'
                
                # 強い逆シグナルが出たら逃げる (0.60以上に設定して簡単には逃げない)
                if side == 'LONG' and down_prob > 0.60:
                    action = 'CLOSE'; reason = 'Strong Reversal'
                elif side == 'SHORT' and up_prob > 0.60:
                    action = 'CLOSE'; reason = 'Strong Reversal'
            
            # === 新規エントリー判定 ===
            elif position is None and balance > 10:
                if vol_pct < MIN_VOLATILITY_PCT: continue 
                
                # ★修正: 期待値フィルターを少し甘くしてエントリーしやすくする
                max_prob = max(up_prob, down_prob)
                if max_prob < BASE_THRESHOLD: continue

                # エントリーシグナル
                if (up_prob >= BASE_THRESHOLD and up_prob > down_prob and rsi < 75):
                    action = 'BUY'
                elif (down_prob >= BASE_THRESHOLD and down_prob > up_prob and rsi > 25):
                    action = 'SELL'
            
            # === 執行 ===
            if action == 'CLOSE' and position:
                sz = position['size']
                if position['side'] == 'LONG': raw_pnl = (price - position['entry_price']) * sz
                else: raw_pnl = (position['entry_price'] - price) * sz
                
                fee = (price * sz) * FEE_RATE
                net_pnl = raw_pnl - fee
                balance += (raw_pnl + (position['entry_price']*sz)) - (position['entry_price']*sz)
                balance += net_pnl
                
                trades.append({'pnl': net_pnl, 'reason': reason})
                position = None
            
            elif action in ['BUY', 'SELL'] and position is None:
                usd_size = balance * 0.95
                sz = usd_size / price
                fee = usd_size * FEE_RATE
                balance -= fee
                position = {'side': 'LONG' if action == 'BUY' else 'SHORT', 'size': sz, 'entry_price': price, 'entry_time': ts}
                ctx = {}

        # 結果表示
        print("\n" + "="*60)
        print("📊 main.py 改良版バックテスト結果")
        print("="*60)
        
        if not trades:
            print("取引なし"); return

        df_trades = pd.DataFrame(trades)
        total_profit = balance - INITIAL_CAPITAL
        win_trades = df_trades[df_trades['pnl'] > 0]
        
        print(f"初期資金: ${INITIAL_CAPITAL:.2f}")
        print(f"最終資金: ${balance:.2f}")
        print(f"純損益  : ${total_profit:.2f} ({total_profit/INITIAL_CAPITAL*100:+.2f}%)")
        print("-" * 30)
        print(f"総取引数: {len(df_trades)}回")
        print(f"勝率    : {len(win_trades)/len(df_trades)*100:.1f}%")
        print("-" * 30)
        print("理由別決済:")
        print(df_trades['reason'].value_counts())
        print("="*60)
        df_trades.to_csv("accurate_backtest_result.csv", index=False)

if __name__ == "__main__":
    tester = AccurateBacktester(CSV_FILE)
    tester.run()