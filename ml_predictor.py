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
        self.lgb_reg_path = f"{model_dir}/lgb_reg_{symbol}.pkl"
        self.lstm_path = f"{model_dir}/lstm_{symbol}.h5"
        
        self.model_lock = threading.Lock()
        
        self.lgb_model = None
        self.lgb_reg_model = None
        self.lstm_model = None
        
        # ★改善点: 特徴量に直近の変化(Lag)とボラティリティ比率を追加
        self.feature_cols = [
            'orderbook_imbalance',  
            'btc_correlation',      
            'btc_trend_strength',
            'rsi', 'macd_hist', 'bb_position', 'bb_width',
            'atr', 'volume_ratio', 'price_change_1h',
            'return_lag_1', 'return_lag_2', 'return_lag_3', 
            'volatility_ratio',
            'sma_20_50_ratio', 'volatility',
            'hour_sin', 'hour_cos', 'day_of_week'
        ]
        self.lstm_lookback = 60
        self.load_models()

    def create_features_from_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """履歴データから特徴量を計算 (推論用)"""
        df = df.copy()
        if len(df) < 100: return None

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
        
        # Price Change & Lags (★改善点)
        # pct_change(1) は1足ごとの変化率
        current_return = close.pct_change(1) * 100
        df['price_change_1h'] = current_return # 互換性のため維持
        df['price_change_4h'] = close.pct_change(4) * 100
        
        # 直近のローソク足の変化率を明示的に特徴量化
        df['return_lag_1'] = current_return.shift(1).fillna(0)
        df['return_lag_2'] = current_return.shift(2).fillna(0)
        df['return_lag_3'] = current_return.shift(3).fillna(0)
        
        # Volatility Ratio (★改善点: 短期ボラ / 長期ボラ)
        # 急激に動き出した瞬間を捉える
        df['atr'] = self._calc_atr(df)
        long_term_atr = df['atr'].rolling(10).mean().replace(0, 1)
        df['volatility_ratio'] = df['atr'] / long_term_atr

        df['volatility'] = close.rolling(20).std() / sma20 * 100
        
        # Time Features
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
            if hasattr(dates, 'dt'):
                hours = dates.dt.hour
                dayofweek = dates.dt.dayofweek
            else:
                hours = dates.hour
                dayofweek = dates.dayofweek
        else:
            dates = df.index
            if hasattr(dates, 'hour'):
                hours = dates.hour
                dayofweek = dates.dayofweek
            else:
                hours = pd.Series(0, index=df.index)
                dayofweek = pd.Series(0, index=df.index)

        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        df['day_of_week'] = dayofweek / 6.0

        available_cols  = [c for c in self.feature_cols if c in df.columns]
        latest_features = df.iloc[[-1]][available_cols].fillna(0)
        
        return latest_features

    def _calc_atr(self, df):
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/14, adjust=False).mean()

    def prepare_lstm_data(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) < self.lstm_lookback + 1:
            return np.zeros((1, self.lstm_lookback, 1))
        s = pd.Series(prices)
        returns = np.log(s / s.shift(1)).fillna(0).values
        window = returns[-self.lstm_lookback:]
        mean = window.mean()
        std = window.std() + 1e-8
        normalized = (window - mean) / std
        return normalized.reshape(1, self.lstm_lookback, 1)

    def predict(self, df: pd.DataFrame, extra_features: dict = None) -> dict:
        if df is None or len(df) < 100:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'データ不足', 'model_used': 'NONE'}

        features = self.create_features_from_history(df)
        if features is None:
            return {'action': 'HOLD', 'confidence': 0, 'model_used': 'NONE'}
        
        if extra_features:
            features['orderbook_imbalance'] = extra_features.get('orderbook_imbalance', 0.0)
            features['btc_correlation'] = extra_features.get('btc_trend', 0.0) 
            features['btc_trend_strength'] = abs(extra_features.get('btc_trend', 0.0))
        else:
            features['orderbook_imbalance'] = 0.0
            features['btc_correlation'] = 0.0
            features['btc_trend_strength'] = 0.0
        
        for col in self.feature_cols:
            if col not in features.columns: features[col] = 0.0
        features = features[self.feature_cols]

        with self.model_lock:
            lgb_model = self.lgb_model
            lstm_model = self.lstm_model
            lgb_reg_model = self.lgb_reg_model

        lgb_up, lgb_down = 0.0, 0.0
        if lgb_model:
            try:
                lgb_pred = lgb_model.predict(features)
                lgb_down = float(lgb_pred[0][0])
                lgb_up = float(lgb_pred[0][2])
            except: pass

        lstm_up, lstm_down = 0.0, 0.0
        if lstm_model:
            try:
                prices = df['close'].values
                inp = self.prepare_lstm_data(prices)
                lstm_pred = lstm_model.predict(inp, verbose=0)[0]
                lstm_down = float(lstm_pred[0])
                lstm_up = float(lstm_pred[2])
            except: pass
        
        # ★改善点: 回帰モデル(Regression)の重要度アップ
        predicted_change_pct = 0.0
        if lgb_reg_model:
            try:
                reg_pred = lgb_reg_model.predict(features)
                predicted_change_pct = float(reg_pred[0])
            except: pass

        if lgb_model and lstm_model:
            final_up = (lgb_up * 0.6 + lstm_up * 0.4)
            final_down = (lgb_down * 0.6 + lstm_down * 0.4)
            model_name = "Ensemble"
        elif lgb_model:
            final_up, final_down = lgb_up, lgb_down
            model_name = "LightGBM"
        elif lstm_model:
            final_up, final_down = lstm_up, lstm_down
            model_name = "LSTM"
        else:
            return {'action': 'HOLD', 'confidence': 0, 'reasoning': 'モデル予測失敗', 'model_used': 'NONE'}

        # ★フィルター緩和: 不均衡が極端な場合のみ除外
        filter_reason = ""
        imbalance = features['orderbook_imbalance'].iloc[-1]
        
        if final_up > final_down:
            if imbalance < -0.8: # -0.6 -> -0.8
                filter_reason = f"売り板厚過多(Imb:{imbalance:.2f})"
                final_up = 0.0 
        elif final_down > final_up:
            if imbalance > 0.8: # 0.6 -> 0.8
                filter_reason = f"買い板厚過多(Imb:{imbalance:.2f})"
                final_down = 0.0

        # Confidence計算
        max_prob = max(final_up, final_down)
        confidence = (max_prob - 0.35) / (0.9 - 0.35) * 100 # 基準を少し下げてスケーリング
        confidence = min(100, max(0, confidence))

        return {
            'action': 'PREDICTED',
            'up_prob': final_up,
            'down_prob': final_down,
            'confidence': int(confidence),
            'model_used': model_name,
            'predicted_change': predicted_change_pct,
            'reasoning': f"Up:{final_up:.2f} Down:{final_down:.2f} PredChange:{predicted_change_pct:+.3f}%",
            'filter_reason': filter_reason
        }

    # (train_regressor, evaluate_model, train_lightgbm, train_lstm, load_models は変更なしでOKですが、
    def train_regressor(self, X, y, X_val=None, y_val=None):
        if not LIGHTGBM_AVAILABLE: return
        print("📊 変動幅予測モデル(Regressor)の学習を開始...")
        params = {'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'random_state': 42, 'learning_rate': 0.05, 'num_leaves': 31}
        train_data = lgb.Dataset(X, label=y)
        valid_sets = []
        if X_val is not None: valid_sets = [lgb.Dataset(X_val, label=y_val, reference=train_data)]
        reg_model = lgb.train(params, train_data, num_boost_round=100, valid_sets=valid_sets)
        with self.model_lock:
            self.lgb_reg_model = reg_model
            joblib.dump(self.lgb_reg_model, self.lgb_reg_path)
    
    def evaluate_model(self, model, X_val, y_val, model_type='lgb'):
        if not SKLEARN_AVAILABLE: return 0.0
        try:
            if len(X_val) == 0: return 0.0
            if model_type == 'lgb':
                preds = model.predict(X_val)
                pred_classes = np.argmax(preds, axis=1)
                y_true = y_val.map({-1:0, 0:1, 1:2}).fillna(1)
                return accuracy_score(y_true, pred_classes)
            return 0.0
        except: return 0.0

    def train_lightgbm(self, X, y, X_val=None, y_val=None):
        if not LIGHTGBM_AVAILABLE: return
        params = {'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss', 'verbose': -1, 'random_state': 42, 'learning_rate': 0.05, 'num_leaves': 31}
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
        X, y = [], []
        s = pd.Series(prices)
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
        model = Sequential([LSTM(64, return_sequences=True, input_shape=(lookback, 1)), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(3, activation='softmax')])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
        with self.model_lock:
            self.lstm_model = model
            model.save(self.lstm_path)

    def load_models(self):
        if os.path.exists(self.lgb_path) and LIGHTGBM_AVAILABLE: self.lgb_model = joblib.load(self.lgb_path)
        if os.path.exists(self.lgb_reg_path) and LIGHTGBM_AVAILABLE: self.lgb_reg_model = joblib.load(self.lgb_reg_path)
        if os.path.exists(self.lstm_path) and KERAS_AVAILABLE: self.lstm_model = keras.models.load_model(self.lstm_path)