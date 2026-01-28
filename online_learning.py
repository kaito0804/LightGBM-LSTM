"""
オンライン学習システム (デイトレード最適化版)
- 15分足データの収集
- モデル更新時の精度チェック機能
"""

import pandas as pd
import os
import threading
import time
from datetime import datetime
from ml_predictor import MLPredictor
from data_collector import DataCollector
import lightgbm as lgb

class OnlineLearner:
    def __init__(self, symbol='ETH', timeframe='15m', retrain_interval_hours=24):
        self.symbol = symbol
        self.timeframe = timeframe # デイトレ用に15mなどを指定可能
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
        """
        最新データを収集してCSVに追記
        """
        # 指定されたtimeframeで収集
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
        """
        安全装置付き再学習プロセス
        """
        # 1. 最新データ収集
        df = self.collect_latest_data(lookback_limit=200)
        if df is None or len(df) < 500:
            print("⚠️ 学習データ不足のためスキップ")
            return

        print(f"🔄 安全再学習開始: {len(df)} lines")
        
        # 2. 直近データを検証用(Validation)に取り分ける (最新15%)
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        feature_cols = self.predictor.feature_cols
        
        # 特徴量の欠損埋め
        for c in feature_cols:
            if c not in train_df.columns: train_df[c] = 0
            if c not in val_df.columns: val_df[c] = 0

        X_val = val_df[feature_cols]
        y_val = val_df['label']
        
        # === LightGBM 安全更新 ===
        # 現在のモデルの精度を確認
        current_acc = 0.0
        with self.predictor.model_lock:
            if self.predictor.lgb_model:
                current_acc = self.predictor.evaluate_model(self.predictor.lgb_model, X_val, y_val, 'lgb')
        
        print(f"   現在のLGBM精度: {current_acc:.4f}")

        # 新規モデル学習
        X_train = train_df[feature_cols]
        y_train = train_df['label']
        y_train_mapped = y_train.map({-1:0, 0:1, 1:2})
        
        params = {
            'objective': 'multiclass', 'num_class': 3, 'verbose': -1, 
            'random_state': 42, 'learning_rate': 0.05
        }
        train_set = lgb.Dataset(X_train, label=y_train_mapped)
        new_lgb = lgb.train(params, train_set, num_boost_round=100)
        
        # 新規モデル評価
        new_acc = self.predictor.evaluate_model(new_lgb, X_val, y_val, 'lgb')
        print(f"   新規LGBM精度: {new_acc:.4f}")
        
        # 更新判定 (精度が悪化していなければ採用)
        if new_acc >= current_acc - 0.03: # 多少のブレは許容
            print("✨ LGBM更新承認")
            with self.predictor.model_lock:
                self.predictor.lgb_model = new_lgb
                try:
                    import joblib
                    joblib.dump(new_lgb, self.predictor.lgb_path)
                except: pass
        else:
            print("🛑 LGBM更新却下: 精度劣化")

        # === LSTM 再学習 (常時更新) ===
        # LSTMは構造上、継続学習に近い形をとるためここではそのまま更新
        print("🧠 LSTM再学習中...")
        if 'close' in df.columns:
            prices = df['close'].values
            labels = df['label'].values
            self.predictor.train_lstm(prices, labels)

        self.last_retrain_time = time.time()
        print(f"✨ モデル更新プロセス完了")
    


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