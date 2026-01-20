"""
機械学習ベースの価格予測システム (修正版)
- LightGBM: テーブルデータに強い勾配ブースティング
- LSTM: 時系列パターン学習
- アンサンブル予測で精度向上
- 自信度計算の最適化
"""
import numpy as np
import pandas as pd
import joblib
import os
import threading

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

class MLPredictor:
    def __init__(self, symbol='ETH', model_dir='models'):
        self.symbol = symbol
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.lgb_path = f"{model_dir}/lgb_{symbol}.pkl"
        self.lstm_path = f"{model_dir}/lstm_{symbol}.h5"
        
        # スレッドロックの初期化
        self.model_lock = threading.Lock()
        
        self.lgb_model = None
        self.lstm_model = None
        
        # 特徴量は学習時(data_collector)と完全に一致させる必要があります
        self.feature_cols = [
            'rsi', 'macd_hist', 'bb_position', 'bb_width',
            'atr', 'volume_ratio', 'price_change_1h',
            'price_change_4h', 'sma_20_50_ratio', 'volatility',
            'hour_sin', 'hour_cos', 'day_of_week'
        ]
        self.lstm_lookback = 60
        self.load_models()

    def create_features_from_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        履歴データ(OHLCV)から特徴量を計算し、予測用の最新1行を返す
        """
        df = df.copy()
        if len(df) < 100:
            return None

        # --- テクニカル指標計算 (DataCollectorとロジック統一) ---
        close = df['close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        # ゼロ除算対策
        loss = loss.replace(0, np.nan)
        rs = gain / loss
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
        
        # SMA & Ratio
        sma50 = close.rolling(50).mean()
        df['sma_20_50_ratio'] = (sma20 / sma50 - 1) * 100
        
        # Volume Ratio
        vol_ma = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / vol_ma.replace(0, 1)
        
        # Price Changes
        df['price_change_1h'] = close.pct_change(1) * 100
        df['price_change_4h'] = close.pct_change(4) * 100
        
        # Volatility
        df['volatility'] = close.rolling(20).std() / sma20 * 100
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

        # 時間特徴量の追加
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = df.index

        if hasattr(dates, 'hour'):
            hours = dates.hour
            dayofweek = dates.dayofweek
        elif hasattr(dates, 'dt'):
            hours = dates.dt.hour
            dayofweek = dates.dt.dayofweek
        else:
            hours = pd.Series(0, index=df.index)
            dayofweek = pd.Series(0, index=df.index)

        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        df['day_of_week'] = dayofweek / 6.0

        # 最新の1行の特徴量のみを抽出
        latest_features = df.iloc[[-1]][self.feature_cols].fillna(0)
        return latest_features

    def prepare_lstm_data(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) < self.lstm_lookback:
            return np.zeros((1, self.lstm_lookback, 1))
        
        window = prices[-self.lstm_lookback:]
        min_p = window.min()
        max_p = window.max()
        
        if max_p - min_p < 1e-8:
            normalized = np.zeros_like(window)
        else:
            normalized = (window - min_p) / (max_p - min_p)
            
        return normalized.reshape(1, self.lstm_lookback, 1)

    def predict(self, df_1h: pd.DataFrame) -> dict:
        """
        DataFrameを受け取り、LGBMとLSTMで予測を行う
        """
        if df_1h is None or len(df_1h) < 100:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'データ不足', 'model_used': 'NONE'}

        # 1. 特徴量作成
        features = self.create_features_from_history(df_1h)
        if features is None:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': '特徴量計算不可', 'model_used': 'NONE'}

        # ロックを取得してモデルを使用
        with self.model_lock:
            lgb_model = self.lgb_model
            lstm_model = self.lstm_model

        # 2. LightGBM 予測
        lgb_up_prob = 0.0
        lgb_down_prob = 0.0
        lgb_used = False
        
        if lgb_model:
            try:
                lgb_pred = lgb_model.predict(features)
                lgb_down_prob = float(lgb_pred[0][0])
                lgb_up_prob = float(lgb_pred[0][2])
                lgb_used = True
            except Exception as e:
                print(f"⚠️ LGBM予測エラー: {e}")

        # 3. LSTM 予測
        lstm_up_prob = 0.0
        lstm_down_prob = 0.0
        lstm_used = False
        
        if lstm_model:
            try:
                prices = df_1h['close'].values
                inp = self.prepare_lstm_data(prices)
                lstm_pred = lstm_model.predict(inp, verbose=0)[0]
                lstm_down_prob = float(lstm_pred[0])
                lstm_up_prob = float(lstm_pred[2])
                lstm_used = True
            except Exception as e:
                print(f"⚠️ LSTM予測エラー: {e}")

        # 4. アンサンブル (平均)
        if lgb_used and lstm_used:
            final_up = (lgb_up_prob + lstm_up_prob) / 2
            final_down = (lgb_down_prob + lstm_down_prob) / 2
            model_name = "Ensemble(LGBM+LSTM)"
        elif lgb_used:
            final_up = lgb_up_prob
            final_down = lgb_down_prob
            model_name = "LightGBM"
        elif lstm_used:
            final_up = lstm_up_prob
            final_down = lstm_down_prob
            model_name = "LSTM"
        else:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'モデル予測失敗', 'model_used': 'NONE'}

        # 自信度計算の改善
        max_prob = max(final_up, final_down)
        
        # 0.35以上あれば「傾向あり」とみなすスケーリング
        if max_prob < 0.35:
            confidence = 0
        else:
            # 0.35 -> 0, 0.85 -> 100 の範囲でスケーリング
            confidence = (max_prob - 0.35) / (0.85 - 0.35) * 100
            confidence = min(100, max(0, confidence))

        return {
            'action': 'PREDICTED',
            'up_prob': final_up,
            'down_prob': final_down,
            'confidence': int(confidence),
            'model_used': model_name,
            'reasoning': f"Up:{final_up:.2f} Down:{final_down:.2f}"
        }

    def train_lightgbm(self, X, y, X_val=None, y_val=None):
        if not LIGHTGBM_AVAILABLE: return
        
        # ロック外で学習
        params = {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'verbose': -1, 'random_state': 42}
        y_mapped = y.map({-1:0, 0:1, 1:2})
        train_data = lgb.Dataset(X, label=y_mapped)
        valid_sets = []
        if X_val is not None:
            y_val_mapped = y_val.map({-1:0, 0:1, 1:2})
            valid_sets = [lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)]
        
        new_model = lgb.train(params, train_data, num_boost_round=100, valid_sets=valid_sets)
        
        # ロックして保存
        with self.model_lock:
            self.lgb_model = new_model
            joblib.dump(self.lgb_model, self.lgb_path)

    def train_lstm(self, prices, labels, lookback=60, epochs=20):
        if not KERAS_AVAILABLE: return
        
        # データ準備
        X, y = [], []
        for i in range(lookback, len(prices)):
            window = prices[i-lookback:i]
            denom = window.max() - window.min()
            if denom < 1e-8: denom = 1
            norm = (window - window.min()) / denom
            X.append(norm)
            l = labels[i]
            if l == -1: enc = [1,0,0]
            elif l == 0: enc = [0,1,0]
            else: enc = [0,0,1]
            y.append(enc)
            
        if len(X) == 0: return

        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        
        # モデル構築・学習
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)), Dropout(0.2),
            LSTM(32), Dropout(0.2), Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
        
        # ロックして保存
        with self.model_lock:
            self.lstm_model = model
            model.save(self.lstm_path)

    def load_models(self):
        if os.path.exists(self.lgb_path) and LIGHTGBM_AVAILABLE:
            try:
                self.lgb_model = joblib.load(self.lgb_path)
            except:
                print("⚠️ LightGBMモデル読み込み失敗 (再学習してください)")
                
        if os.path.exists(self.lstm_path) and KERAS_AVAILABLE:
            try:
                self.lstm_model = keras.models.load_model(self.lstm_path)
            except:
                print("⚠️ LSTMモデル読み込み失敗 (再学習してください)")


# ===== 使用例 (修正版) =====
if __name__ == "__main__":
    print("="*70)
    print("🤖 機械学習予測システムテスト")
    print("="*70)
    
    predictor = MLPredictor('ETH')
    
    # ランダムなダミーデータでテスト
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    dummy_data = {
        'timestamp': dates,
        'open': np.random.rand(200) * 100 + 3000,
        'high': np.random.rand(200) * 100 + 3050,
        'low': np.random.rand(200) * 100 + 2950,
        'close': np.random.rand(200) * 100 + 3000,
        'volume': np.random.rand(200) * 1000
    }
    df = pd.DataFrame(dummy_data)
    
    print("\n📊 予測実行中...")
    result = predictor.predict(df)
    
    print(f"✅ 結果: {result['action']} (信頼度: {result['confidence']}%)")
    print(f"   詳細: {result['reasoning']}")