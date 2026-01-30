"""
オンライン学習システム (デイトレード最適化版・修正版)
- 分類モデル(方向)と回帰モデル(値幅)の両方を安全に更新
- 期待値ロジックの鮮度を維持する
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
        
        self.collector = DataCollector(symbol)
        self.predictor = MLPredictor(symbol)
        
        self.training_data_path = f"training_data/{symbol}_{timeframe}_training.csv"
        self.last_retrain_time = time.time()
        self.max_rows = 40000
        
        self.learning_thread = None
        self.is_running = False
        
        print(f"🔄 オンライン学習初期化: {timeframe}足 (間隔: {retrain_interval_hours}h)")
    
    def collect_latest_data(self, lookback_limit=500):
        """最新データを収集してCSVに追記"""
        new_df = self.collector.collect_historical_data(timeframe=self.timeframe, limit=lookback_limit)
        
        if new_df is None or new_df.empty:
            return None

        if os.path.exists(self.training_data_path):
            try:
                existing_df = pd.read_csv(self.training_data_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            except:
                combined_df = new_df
        else:
            combined_df = new_df

        if len(combined_df) > self.max_rows:
            combined_df = combined_df.tail(self.max_rows)

        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        combined_df.to_csv(self.training_data_path, index=False)
        
        return combined_df
    
    def retrain_models(self):
        """安全装置付き再学習プロセス (分類 & 回帰)"""
        # 1. 最新データ収集
        df = self.collect_latest_data(lookback_limit=300)
        if df is None or len(df) < 500:
            print("⚠️ 学習データ不足のためスキップ")
            return

        print(f"🔄 安全再学習開始: {len(df)} lines")
        
        # 2. 直近データを検証用(Validation)に取り分ける
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        feature_cols = self.predictor.feature_cols
        
        # 特徴量の整合性チェック & 欠損埋め
        # (重要: data_collector.pyが修正されていないとここで0埋めになり性能が落ちる)
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