import os
import time
import pandas as pd
import numpy as np
import ccxt
import joblib
import tensorflow as tf
import logging

# === ログ・警告の完全消去設定 ===
import warnings
# 特定の警告メッセージを無視
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# TensorFlowのログも消す
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from datetime import datetime, timedelta
# tqdm（バー）を無効化してインポート
from tqdm import tqdm as original_tqdm
def tqdm(*args, **kwargs):
    kwargs['disable'] = True # バーを強制非表示
    return original_tqdm(*args, **kwargs)

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# === 設定エリア (トレンドフォロー版) ===
SYMBOL = 'ETH/USDT'
TIMEFRAME = '15m'
SPLIT_DATE = "2025-01-01 00:00:00" 
FETCH_DAYS = 730

# ロジック設定
INITIAL_BALANCE = 500

# ★変更1: エントリーは厳選する
ENTRY_THRESHOLD = 0.60 
CONFIDENCE_THRESHOLD = 60 

# ★変更2: 撤退ラインを「0.55」に引き上げ
#  「反対方向に行く確率が55%を超えたら」初めて逃げる。
#  (50%前後の迷っている状態なら、ポジションを握り続ける)
CLOSE_THRESHOLD = 0.55 

FEE_RATE = 0.00035

class StrictBacktesterFixed:
    def __init__(self):
        self.exchange = ccxt.binance()
        self.scaler = StandardScaler()
        self.lgb_model = None
        self.lstm_model = None
        
        self.balance = INITIAL_BALANCE
        self.position = None
        self.entry_price = 0
        self.position_size = 0
        self.entry_fee_cost = 0 # エントリー時の手数料を記憶
        self.trades = []

    def fetch_data(self):
        # 保存するファイル名
        filename = f"backtest_data_{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
        
        # 1. すでにファイルがあるかチェック
        if os.path.exists(filename):
            print(f"📂 保存済みデータ ({filename}) を読み込み中...")
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # データの鮮度チェック（オプション）
            last_date = df['timestamp'].iloc[-1]
            print(f"   データ期間: {df['timestamp'].iloc[0]} 〜 {last_date}")
            return df
        
        # 2. ファイルがない場合はダウンロード
        print(f"📥 過去 {FETCH_DAYS} 日分のデータを新規取得中...")
        since = self.exchange.parse8601((datetime.now() - timedelta(days=FETCH_DAYS)).strftime('%Y-%m-%d %H:%M:%S'))
        all_candles = []
        pbar = tqdm(total=int(FETCH_DAYS * 24 * 4))
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
                if not candles: break
                since = candles[-1][0] + 1
                all_candles += candles
                pbar.update(len(candles))
                if candles[-1][0] > time.time() * 1000: break
                time.sleep(0.1)
            except:
                break
        pbar.close()
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # 3. CSVに保存
        df.to_csv(filename, index=False)
        print(f"💾 データを {filename} に保存しました（次回から高速化されます）")
        
        return df

    def add_features(self, df):
        df = df.copy()
        df['return'] = df['close'].pct_change()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean() # 追加
        df['volatility'] = df['return'].rolling(20).std()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 正解ラベル: 次の足のCloseが今のCloseより高いか
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df.dropna()

    def train_models(self, train_df):
        print("\n🧠 2024年以前のデータでモデルを学習中...")
        features = ['close', 'volume', 'sma_20', 'sma_50', 'volatility', 'rsi']
        X = train_df[features].values
        y = train_df['target'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        print("   Training LightGBM...")
        self.lgb_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
        self.lgb_model.fit(X_scaled, y)
        
        print("   Training LSTM...")
        X_lstm = []
        y_lstm = []
        lookback = 60
        for i in range(lookback, len(X_scaled)):
            X_lstm.append(X_scaled[i-lookback:i])
            y_lstm.append(y[i])
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)
        
        model = Sequential([
            LSTM(64, return_sequences=False, input_shape=(lookback, len(features))),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
        # Epochを増やしてしっかり学習
        model.fit(X_lstm, y_lstm, epochs=10, batch_size=64, verbose=0)
        self.lstm_model = model
        print("✅ 学習完了")

    def run_backtest(self):
        # 1. データ準備
        full_df = self.fetch_data()
        full_df = self.add_features(full_df)
        
        split_ts = pd.to_datetime(SPLIT_DATE)
        train_df = full_df[full_df['timestamp'] < split_ts].copy()
        test_df = full_df[full_df['timestamp'] >= split_ts].copy()
        
        print(f"\n📊 データ分割結果: 学習 {len(train_df)}件 / テスト {len(test_df)}件")
        self.train_models(train_df)
        
        print(f"\n🚀 {SPLIT_DATE} 以降のデータでバックテスト開始...")
        print("⚡ 推論を高速化処理中 (数万件を一括計算します)...")
        
        features = ['close', 'volume', 'sma_20', 'sma_50', 'volatility', 'rsi']
        
        # テスト用に、直前のデータ(60本)を含めて結合
        combined_df = pd.concat([train_df.tail(60), test_df]).reset_index(drop=True)
        combined_features = self.scaler.transform(combined_df[features].values)
        
        # --- 高速化: 事前に全データを推論する ---
        # LSTM用のデータセットを一気に作成
        X_lstm = []
        # テスト対象となるインデックス(60番目)から最後まで
        for i in range(60, len(combined_df)):
            X_lstm.append(combined_features[i-60:i])
        X_lstm = np.array(X_lstm)
        
        # LightGBM用のデータセット
        X_lgb = combined_features[60:]
        
        # ★ここで一括推論 (1回だけ実行するので爆速)
        lgb_probs = self.lgb_model.predict_proba(X_lgb)[:, 1]
        lstm_probs = self.lstm_model.predict(X_lstm, batch_size=4096, verbose=0)[:, 0]
        
        print("✅ 一括推論完了。シミュレーションを実行します...")

        # ループ開始 (推論済みの確率を使って判定のみ行う)
        # lgb_probs の長さは test_df と同じ
        for i in tqdm(range(len(lgb_probs))):
            # combined_df 上のインデックスは +60
            current_idx = i + 60
            row = combined_df.iloc[current_idx]
            timestamp = row['timestamp']
            
            if timestamp < split_ts: continue

            price = row['close']
            volatility = row['volatility']
            
            # ★ 事前計算した確率を取り出すだけ (計算コストゼロ)
            lgb_prob = lgb_probs[i]
            lstm_prob = lstm_probs[i]
            
            up_prob = (lgb_prob + lstm_prob) / 2
            down_prob = 1.0 - up_prob
            confidence = max(up_prob, down_prob) * 100
            
            # --- ロジック実行 ---
            if volatility > 0.03: sl_pct, tp_pct = 0.02, 0.06
            else: sl_pct, tp_pct = 0.015, 0.03 # ★変更: SLを少し広げる (0.01 -> 0.015)
                
            action = 'HOLD'
            reason = ""
            
            # 各種テクニカル指標の取得
            sma_50_val = row['sma_50']
            rsi_val = row['rsi'] # ★追加
            
            # 決済判定
            if self.position == 'LONG':
                pnl_pct = (price - self.entry_price) / self.entry_price
                if down_prob > CLOSE_THRESHOLD: action = 'CLOSE'; reason = "AI撤退"
                elif pnl_pct <= -sl_pct: action = 'CLOSE'; reason = "損切り"
                elif pnl_pct >= tp_pct: action = 'CLOSE'; reason = "利確"
            
            elif self.position == 'SHORT':
                pnl_pct = (self.entry_price - price) / self.entry_price
                if up_prob > CLOSE_THRESHOLD: action = 'CLOSE'; reason = "AI撤退"
                elif pnl_pct <= -sl_pct: action = 'CLOSE'; reason = "損切り"
                elif pnl_pct >= tp_pct: action = 'CLOSE'; reason = "利確"

            # 新規エントリー
            if self.position is None and self.balance > 10:
                if confidence >= CONFIDENCE_THRESHOLD:
                    # ★修正: RSIフィルターを追加
                    # 上昇予測 & SMAより上 & 「買われすぎ(70)ではない」
                    if up_prob >= ENTRY_THRESHOLD and price > sma_50_val and rsi_val < 70:
                        action = 'BUY'
                    # 下落予測 & SMAより下 & 「売られすぎ(30)ではない」
                    elif down_prob >= ENTRY_THRESHOLD and price < sma_50_val and rsi_val > 30:
                        action = 'SELL'
            
            # 実行
            if action == 'BUY' and self.position is None:
                self._entry('LONG', price, timestamp)
            elif action == 'SELL' and self.position is None:
                self._entry('SHORT', price, timestamp)
            elif action == 'CLOSE' and self.position is not None:
                self._close(price, timestamp, reason)

        self._print_result()

    def _entry(self, side, price, timestamp):
        self.position = side
        self.entry_price = price
        self.position_size = (self.balance / price)
        fee = self.balance * FEE_RATE
        self.entry_fee_cost = fee # ★手数料を記録
        self.balance -= fee
    
    def _close(self, price, timestamp, reason):
        value = self.position_size * price
        raw_pnl = 0
        if self.position == 'LONG': raw_pnl = value - (self.position_size * self.entry_price)
        else: raw_pnl = (self.position_size * self.entry_price) - value
        
        exit_fee = value * FEE_RATE
        # ★純損益 = 粗利 - (エントリー手数料 + 決済手数料)
        net_pnl = raw_pnl - exit_fee - self.entry_fee_cost
        
        self.balance += (raw_pnl - exit_fee)
        
        self.trades.append({
            'time': timestamp, 
            'pnl': net_pnl, # ★正しい純損益を記録
            'reason': reason, 
            'balance': self.balance
        })
        self.position = None
        self.position_size = 0
        self.entry_fee_cost = 0

    def _print_result(self):
        print("\n" + "="*50)
        print("📊 修正版・厳密なバックテスト結果")
        print(f"   期間: {SPLIT_DATE} 〜 現在")
        print("="*50)
        if not self.trades:
            print("取引なし")
            return
            
        df = pd.DataFrame(self.trades)
        wins = df[df['pnl'] > 0]
        total = len(df)
        if total > 0:
            win_rate = len(wins) / total * 100
        else:
            win_rate = 0
            
        profit = df['pnl'].sum()
        
        print(f"初期資金: ${INITIAL_BALANCE}")
        print(f"最終資金: ${self.balance:.2f}")
        print(f"純損益: ${profit:.2f}")
        print(f"勝率: {win_rate:.2f}% ({len(wins)}/{total})")
        print(f"取引回数: {total}回")
        print("-" * 50)
        print(df['reason'].value_counts())
        print("="*50)

if __name__ == "__main__":
    tester = StrictBacktesterFixed()
    tester.run_backtest()