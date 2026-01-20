"""
オンライン学習システム
- 稼働中にモデルを定期的に再学習
- 新しいデータを追加して精度向上
- バックグラウンドで実行
"""

import pandas as pd
import os
import threading
import time
from datetime import datetime
from ml_predictor import MLPredictor
from data_collector import DataCollector

class OnlineLearner:
    """
    オンライン学習マネージャー
    - 定期的にデータを収集
    - モデルを再学習
    - 精度をモニタリング
    """
    
    def __init__(self, symbol='ETH', retrain_interval_hours=24):
        self.symbol = symbol
        self.retrain_interval = retrain_interval_hours * 3600  # 秒に変換
        
        # コンポーネント初期化
        self.collector = DataCollector(symbol)
        self.predictor = MLPredictor(symbol)
        
        self.training_data_path = f"training_data/{symbol}_1h_training.csv"
        self.last_retrain_time = time.time()
        self.max_rows = 5000  # 保持する最大行数
        
        # 学習スレッド管理
        self.learning_thread = None
        self.is_running = False
        
        print(f"🔄 オンライン学習システム初期化")
        print(f"   再学習間隔: {retrain_interval_hours}時間")
    
    def collect_latest_data(self, lookback_hours=500):
        """
        最新の特徴量付きデータを取得し、既存のCSVに継ぎ足して保存する
        """
        print(f"📊 データ更新・蓄積中: {self.symbol}...")
        
        # DataCollectorを使って特徴量とラベル付きのデータを取得
        # DataCollector内でテクニカル指標計算とラベル付けは完了している前提
        new_df = self.collector.collect_historical_data(timeframe='1h', limit=lookback_hours)
        
        if new_df is None or new_df.empty:
            print("⚠️ 最新データの取得に失敗しました")
            return None

        # 既存データの読み込みと結合
        if os.path.exists(self.training_data_path):
            try:
                existing_df = pd.read_csv(self.training_data_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # 重複削除（timestampを基準に最新を保持）
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
            except Exception as e:
                print(f"⚠️ 既存データの読み込み失敗、新規作成します: {e}")
                combined_df = new_df
        else:
            combined_df = new_df

        # データ量制限 (古すぎるデータは捨てる)
        if len(combined_df) > self.max_rows:
            combined_df = combined_df.tail(self.max_rows)

        # 保存
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        combined_df.to_csv(self.training_data_path, index=False)
        
        print(f"✅ データ蓄積完了: 合計 {len(combined_df)} 行")
        return combined_df
    
    def retrain_models(self):
        """
        蓄積された全データを使って再学習
        """
        # 最新データを取得してから学習
        df = self.collect_latest_data(lookback_hours=100)
        
        if df is None or len(df) < 200:
            print("⚠️ 学習に必要なデータが足りません（最低200行必要）")
            return

        print(f"🔄 再学習フェーズ開始: {len(df)} サンプルを使用")
        
        # 特徴量カラム（MLPredictorの設定に合わせる）
        feature_cols = self.predictor.feature_cols
        
        # カラム存在チェック
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            print(f"❌ 特徴量が不足しています: {missing_cols}")
            # 不足している場合は0で埋める（緊急避難）
            for c in missing_cols:
                df[c] = 0
        
        X = df[feature_cols]
        y = df['label']

        # --- LightGBM 再学習 ---
        print("⚡ LightGBM 再学習中...")
        # ✅ 修正: 正しいメソッド名を使用
        self.predictor.train_lightgbm(X, y)

        # --- LSTM 再学習 ---
        print("🧠 LSTM 再学習中...")
        if 'close' in df.columns and 'label' in df.columns:
            prices = df['close'].values
            labels = df['label'].values
            self.predictor.train_lstm(prices, labels)
        else:
            print("⚠️ LSTM学習に必要なカラム(close, label)が不足しています")

        self.last_retrain_time = time.time()
        print(f"✨ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} モデル更新完了")
    
    def start_background_learning(self):
        """
        バックグラウンドで定期再学習を開始
        """
        if self.is_running:
            print("⚠️ すでに学習スレッドが実行中です")
            return
        
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        print(f"✅ バックグラウンド学習スレッド開始")
    
    def _learning_loop(self):
        """
        学習ループ（別スレッド）
        """
        while self.is_running:
            # 次回学習までの残り時間
            elapsed = time.time() - self.last_retrain_time
            remaining = self.retrain_interval - elapsed
            
            if remaining <= 0:
                # 再学習実行
                try:
                    self.retrain_models()
                except Exception as e:
                    print(f"❌ 再学習エラー: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ✅ 修正: スリープ時間を最適化 (最大でも1時間、残り時間が少なければそれに合わせる)
            sleep_time = min(3600, max(60, remaining))
            time.sleep(sleep_time)
    
    def stop_background_learning(self):
        """バックグラウンド学習停止"""
        self.is_running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        print("⏸️ バックグラウンド学習停止")


# ===== 使用例 =====
if __name__ == "__main__":
    print("="*70)
    print("🔄 オンライン学習システムテスト")
    print("="*70)
    
    try:
        learner = OnlineLearner('ETH', retrain_interval_hours=24)
        print("✅ 初期化成功")
        
        # テスト実行（コメントアウト解除で実際に学習可能）
        # learner.retrain_models()
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()