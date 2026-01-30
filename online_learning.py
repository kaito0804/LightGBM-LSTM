# online_learning.py (修正版)
"""
オンライン学習システム (マルチタイムフレーム & 新特徴量対応版)
- 分類モデル(方向)と回帰モデル(値幅)の両方を安全に更新
- 期待値ロジックの鮮度を維持する
- 15m/1h 両対応
"""

import pandas as pd
import os
import threading
import time
import numpy as np
from datetime import datetime
from ml_predictor import MLPredictor
from data_collector import DataCollector
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

class OnlineLearner:
    def __init__(self, symbol='ETH', timeframe='15m', retrain_interval_hours=24):
        self.symbol = symbol
        self.timeframe = timeframe 
        self.retrain_interval = retrain_interval_hours * 3600
        
        # データ収集器
        self.collector = DataCollector(symbol)
        
        # ★修正1: MLPredictorにtimeframeを渡す
        self.predictor = MLPredictor(symbol, timeframe=timeframe)
        
        self.training_data_path = f"training_data/{symbol}_{timeframe}_training.csv"
        self.last_retrain_time = time.time()
        self.max_rows = 40000
        
        self.learning_thread = None
        self.is_running = False
        
        print(f"🔄 オンライン学習初期化: {timeframe}足 (間隔: {retrain_interval_hours}h)")
    
    def _calculate_features_online(self, df):
        """
        オンライン学習用に特徴量を計算する
        (fetch_binance_data.py のロジックと整合性を取る)
        ※BTCデータはリアルタイム取得が難しいため、相関系は0埋めまたは既存値を維持
        """
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # --- 基本指標 ---
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
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
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        # SMA & Volume
        df['sma_20'] = sma20
        df['sma_50'] = close.rolling(50).mean()
        df['sma_20_50_ratio'] = (df['sma_20'] / df['sma_50'] - 1) * 100
        
        vol_ma = volume.rolling(20).mean()
        df['volume_ratio'] = volume / vol_ma.replace(0, 1)
        
        # --- ★修正2: 新機能の特徴量を追加 ---
        current_return = close.pct_change(1).fillna(0) * 100
        df['price_change_1h'] = current_return
        df['price_change_4h'] = close.pct_change(4).fillna(0) * 100
        
        df['return_lag_1'] = current_return.shift(1).fillna(0)
        df['return_lag_2'] = current_return.shift(2).fillna(0)
        df['return_lag_3'] = current_return.shift(3).fillna(0)
        
        long_term_atr = df['atr'].rolling(10).mean().replace(0, 1)
        df['volatility_ratio'] = df['atr'] / long_term_atr
        df['volatility'] = close.rolling(20).std() / sma20 * 100
        
        # 時間特徴量
        if 'timestamp' in df.columns:
            # timestampがdatetime型でない場合は変換
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
            df['day_of_week'] = df['timestamp'].dt.dayofweek / 6.0
        
        # 不足しているカラム(BTC系など)は0で埋める
        for col in self.predictor.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        # --- ラベル作成 (正解データ) ---
        horizon = 1
        future_change = close.shift(-horizon).pct_change(1) * 100
        df['future_change'] = (df['close'].shift(-horizon) - df['close']) / df['close'] * 100
        
        atr_pct = (df['atr'] / close) * 100
        threshold = (atr_pct * 0.20).clip(0.08, 1.2)
        
        conditions = [
            (df['future_change'] > threshold),
            (df['future_change'] < -threshold)
        ]
        choices = [1, -1] # Buy, Sell
        df['label'] = np.select(conditions, choices, default=0)
        
        return df.dropna()

    def collect_latest_data(self, lookback_limit=500):
        """最新データを収集して特徴量を計算し、CSVに追記"""
        # 1. 生データの収集
        raw_df = self.collector.collect_historical_data(timeframe=self.timeframe, limit=lookback_limit)
        
        if raw_df is None or raw_df.empty:
            return None

        # ★修正3: 保存前に特徴量計算を行う (これがないとcsvが壊れる)
        new_df = self._calculate_features_online(raw_df)

        # 既存データとの結合
        if os.path.exists(self.training_data_path):
            try:
                existing_df = pd.read_csv(self.training_data_path)
                # timestampの型合わせ
                if 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            except Exception as e:
                print(f"⚠️ CSV読み込みエラー(新規作成します): {e}")
                combined_df = new_df
        else:
            combined_df = new_df

        # サイズ制限
        if len(combined_df) > self.max_rows:
            combined_df = combined_df.tail(self.max_rows)

        # 保存
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        combined_df.to_csv(self.training_data_path, index=False)
        
        return combined_df
    
    def retrain_models(self):
        """安全装置付き再学習プロセス"""
        # 1. 最新データ収集 (特徴量計算込み)
        df = self.collect_latest_data(lookback_limit=300)
        if df is None or len(df) < 500:
            print("⚠️ 学習データ不足のためスキップ")
            return

        print(f"🔄 安全再学習開始 ({self.timeframe}): {len(df)} lines")
        
        # 2. 直近データを検証用(Validation)に取り分ける
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        feature_cols = self.predictor.feature_cols
        
        # 特徴量の整合性チェック
        for c in feature_cols:
            if c not in train_df.columns: train_df[c] = 0.0
            if c not in val_df.columns: val_df[c] = 0.0

        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        
        # ==========================================
        # 1. 分類モデル (LightGBM Classifier) の更新
        # ==========================================
        print("📊 [1/3] 分類モデル(方向)の更新チェック")
        y_cls_train = train_df['label']
        y_cls_val = val_df['label']
        
        # 現在の精度確認
        current_acc = 0.0
        with self.predictor.model_lock:
            if self.predictor.lgb_model:
                current_acc = self.predictor.evaluate_model(self.predictor.lgb_model, X_val, y_cls_val, 'lgb')
        
        # 新規学習
        y_train_mapped = y_cls_train.map({-1:0, 0:1, 1:2})
        params_cls = {
            'objective': 'multiclass', 'num_class': 3, 'verbose': -1, 
            'random_state': 42, 'learning_rate': 0.05
        }
        train_set_cls = lgb.Dataset(X_train, label=y_train_mapped)
        new_lgb_cls = lgb.train(params_cls, train_set_cls, num_boost_round=100)
        
        new_acc = self.predictor.evaluate_model(new_lgb_cls, X_val, y_cls_val, 'lgb')
        
        if new_acc >= current_acc - 0.03:
            print(f"   ✅ 更新承認 (Acc: {current_acc:.3f} -> {new_acc:.3f})")
            with self.predictor.model_lock:
                self.predictor.lgb_model = new_lgb_cls
                try:
                    import joblib
                    joblib.dump(new_lgb_cls, self.predictor.lgb_path)
                except: pass
        else:
            print(f"   🛑 更新却下 (精度低下: {current_acc:.3f} -> {new_acc:.3f})")

        # ==========================================
        # 2. 回帰モデル (LightGBM Regressor) の更新
        # ==========================================
        print("📊 [2/3] 回帰モデル(値幅)の更新チェック")
        if 'future_change' in train_df.columns:
            y_reg_train = train_df['future_change']
            y_reg_val = val_df['future_change']
            
            # 現在の誤差(RMSE)確認
            current_rmse = 999.0
            with self.predictor.model_lock:
                if self.predictor.lgb_reg_model:
                    try:
                        preds = self.predictor.lgb_reg_model.predict(X_val)
                        current_rmse = np.sqrt(mean_squared_error(y_reg_val, preds))
                    except: pass
            
            # 新規学習
            params_reg = {
                'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 
                'random_state': 42, 'learning_rate': 0.05
            }
            train_set_reg = lgb.Dataset(X_train, label=y_reg_train)
            new_lgb_reg = lgb.train(params_reg, train_set_reg, num_boost_round=100)
            
            # 新規RMSE確認
            new_preds = new_lgb_reg.predict(X_val)
            new_rmse = np.sqrt(mean_squared_error(y_reg_val, new_preds))
            
            # RMSEは低い方が良い (多少の悪化は許容して最新トレンドに追従させる)
            if new_rmse <= current_rmse * 1.1: 
                print(f"   ✅ 更新承認 (RMSE: {current_rmse:.4f} -> {new_rmse:.4f})")
                with self.predictor.model_lock:
                    self.predictor.lgb_reg_model = new_lgb_reg
                    try:
                        import joblib
                        joblib.dump(new_lgb_reg, self.predictor.lgb_reg_path)
                    except: pass
            else:
                print(f"   🛑 更新却下 (誤差増大: {current_rmse:.4f} -> {new_rmse:.4f})")
        else:
            print("   ⚠️ future_change列がないため回帰モデル学習スキップ")

        # ==========================================
        # 3. LSTM 再学習
        # ==========================================
        print("🧠 [3/3] LSTM再学習中...")
        if 'close' in df.columns:
            prices = df['close'].values
            labels = df['label'].values
            self.predictor.train_lstm(prices, labels)

        self.last_retrain_time = time.time()
        print(f"✨ 全モデル更新プロセス完了")

    def start_background_learning(self):
        if self.is_running: return
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        print(f"✅ バックグラウンド学習開始")
    
    def _learning_loop(self):
        while self.is_running:
            elapsed = time.time() - self.last_retrain_time
            remaining = self.retrain_interval - elapsed
            
            if remaining <= 0:
                try:
                    self.retrain_models()
                except Exception as e:
                    print(f"❌ 再学習エラー: {e}")
            
            sleep_time = min(3600, max(60, remaining))
            time.sleep(sleep_time)

    def stop_background_learning(self):
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)