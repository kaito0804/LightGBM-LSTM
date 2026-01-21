"""
機械学習ベースの価格予測システム (デイトレード最適化版)
- LightGBM: テーブルデータ予測 (板情報追加)
- LSTM: 対数変化率を使用した時系列予測
- 評価機能: オンライン学習の安全性確保
"""
import numpy as np
import pandas as pd
import joblib
import os
import threading

try:
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learnがインストールされていません。'pip install scikit-learn' を実行してください。")

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
        
        self.model_lock = threading.Lock()
        
        self.lgb_model = None
        self.lstm_model = None
        
        # 特徴量定義 (Imbalanceを追加)
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
        履歴データから特徴量を計算 (推論用)
        """
        df = df.copy()
        if len(df) < 100:
            return None

        # テクニカル指標計算
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
        
        # SMA Ratio
        sma50 = close.rolling(50).mean()
        df['sma_20_50_ratio'] = (sma20 / sma50 - 1) * 100
        
        # Volume
        vol_ma = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / vol_ma.replace(0, 1)
        
        # Price Change
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

        # Time Features
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
        else:
            dates = df.index
        
        # タイムスタンプ型に応じた処理
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

        latest_features = df.iloc[[-1]][[c for c in self.feature_cols if c != 'orderbook_imbalance']].fillna(0)
        return latest_features

    def prepare_lstm_data(self, prices: np.ndarray) -> np.ndarray:
        """
        LSTM用データ作成 (対数変化率 + 正規化)
        """
        if len(prices) < self.lstm_lookback + 1:
            return np.zeros((1, self.lstm_lookback, 1))
        
        # 価格そのものではなく、変化率を使う（価格水準が変わっても対応可能に）
        s = pd.Series(prices)
        returns = np.log(s / s.shift(1)).fillna(0).values
        
        window = returns[-self.lstm_lookback:]
        
        # Z-score正規化
        mean = window.mean()
        std = window.std() + 1e-8
        normalized = (window - mean) / std
            
        return normalized.reshape(1, self.lstm_lookback, 1)

    def predict(self, df: pd.DataFrame, extra_features: dict = None) -> dict:
        """
        予測実行 (外部特徴量対応)
        """
        if df is None or len(df) < 100:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'データ不足', 'model_used': 'NONE'}

        # 1. 特徴量作成
        features = self.create_features_from_history(df)
        if features is None:
            return {'action': 'HOLD', 'confidence': 0, 'model_used': 'NONE'}

        # カラム順序の保証と欠損埋め
        for col in self.feature_cols:
            if col not in features.columns:
                features[col] = 0.0
        features = features[self.feature_cols]

        with self.model_lock:
            lgb_model = self.lgb_model
            lstm_model = self.lstm_model

        # 2. LightGBM 予測
        lgb_up = 0.0
        lgb_down = 0.0
        lgb_used = False
        
        if lgb_model:
            try:
                lgb_pred = lgb_model.predict(features)
                lgb_down = float(lgb_pred[0][0])
                lgb_up = float(lgb_pred[0][2])
                lgb_used = True
            except Exception as e:
                print(f"⚠️ LGBM予測エラー: {e}")

        # 3. LSTM 予測
        lstm_up = 0.0
        lstm_down = 0.0
        lstm_used = False
        
        if lstm_model:
            try:
                prices = df['close'].values
                inp = self.prepare_lstm_data(prices)
                lstm_pred = lstm_model.predict(inp, verbose=0)[0]
                lstm_down = float(lstm_pred[0])
                lstm_up = float(lstm_pred[2])
                lstm_used = True
            except Exception as e:
                print(f"⚠️ LSTM予測エラー: {e}")

        # 4. アンサンブル
        if lgb_used and lstm_used:
            final_up = (lgb_up * 0.6 + lstm_up * 0.4) # LGBMを少し重視
            final_down = (lgb_down * 0.6 + lstm_down * 0.4)
            model_name = "Ensemble"
        elif lgb_used:
            final_up = lgb_up
            final_down = lgb_down
            model_name = "LightGBM"
        elif lstm_used:
            final_up = lstm_up
            final_down = lstm_down
            model_name = "LSTM"
        else:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'モデル予測失敗', 'model_used': 'NONE'}

        # 自信度計算
        max_prob = max(final_up, final_down)
        # 0.35以上で反応開始
        if max_prob < 0.35:
            confidence = 0
        else:
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

    def evaluate_model(self, model, X_val, y_val, model_type='lgb'):
        """
        モデルの精度評価 (オンライン学習用)
        """
        if not SKLEARN_AVAILABLE: return 0.0
        try:
            if len(X_val) == 0: return 0.0
            
            if model_type == 'lgb':
                preds = model.predict(X_val)
                pred_classes = np.argmax(preds, axis=1)
                # ラベルマップ: -1->0, 0->1, 1->2
                y_true = y_val.map({-1:0, 0:1, 1:2}).fillna(1)
                return accuracy_score(y_true, pred_classes)
            
            return 0.0
        except Exception as e:
            print(f"評価エラー: {e}")
            return 0.0

    def train_lightgbm(self, X, y, X_val=None, y_val=None):
        if not LIGHTGBM_AVAILABLE: return
        
        # 学習パラメータ (デイトレ用: やや過学習を防ぐ設定)
        params = {
            'objective': 'multiclass', 
            'num_class': 3, 
            'metric': 'multi_logloss', 
            'verbose': -1, 
            'random_state': 42,
            'learning_rate': 0.05,
            'num_leaves': 31
        }
        y_mapped = y.map({-1:0, 0:1, 1:2})
        train_data = lgb.Dataset(X, label=y_mapped)
        valid_sets = []
        if X_val is not None:
            y_val_mapped = y_val.map({-1:0, 0:1, 1:2})
            valid_sets = [lgb.Dataset(X_val, label=y_val_mapped, reference=train_data)]
        
        new_model = lgb.train(params, train_data, num_boost_round=100, valid_sets=valid_sets)
        
        with self.model_lock:
            self.lgb_model = new_model
            joblib.dump(self.lgb_model, self.lgb_path)
    
    def train_lstm(self, prices, labels, lookback=60, epochs=20):
        if not KERAS_AVAILABLE: return
        
        # データ作成
        X, y = [], []
        s = pd.Series(prices)
        # 対数変化率
        returns = np.log(s / s.shift(1)).fillna(0).values
        
        for i in range(lookback, len(returns)):
            window = returns[i-lookback:i]
            mean = window.mean()
            std = window.std() + 1e-8
            norm = (window - mean) / std
            
            X.append(norm)
            l = labels[i]
            if l == -1: enc = [1,0,0]
            elif l == 0: enc = [0,1,0]
            else: enc = [0,0,1]
            y.append(enc)
            
        if len(X) == 0: return

        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)), Dropout(0.2),
            LSTM(32), Dropout(0.2), Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
        
        with self.model_lock:
            self.lstm_model = model
            model.save(self.lstm_path)

    def load_models(self):
        if os.path.exists(self.lgb_path) and LIGHTGBM_AVAILABLE:
            try: self.lgb_model = joblib.load(self.lgb_path)
            except Exception as e: print(f"⚠️ LGBM読み込みエラー: {e}")

        if os.path.exists(self.lstm_path) and KERAS_AVAILABLE:
            try: self.lstm_model = keras.models.load_model(self.lstm_path)
            except Exception as e: print(f"⚠️ LSTM読み込みエラー: {e}")