# train_models.py (統合版)
import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from ml_predictor import MLPredictor

class ModelTrainer:
    def __init__(self, symbol='ETH', timeframe='15m'):
        self.symbol = symbol
        self.timeframe = timeframe
        # MLPredictorにtimeframeを渡し、保存ファイル名を自動で切り替えさせる
        self.predictor = MLPredictor(symbol, timeframe=timeframe)
        
    def train(self):
        print(f"\n{'='*60}")
        print(f"🚀 モデル学習開始: {self.symbol} [{self.timeframe}]")
        print(f"{'='*60}")
        
        # 1. データの特定
        pattern = f"training_data/{self.symbol}_{self.timeframe}_training*.csv"
        files = glob.glob(pattern)
        if not files:
            print(f"❌ エラー: {self.timeframe}用の学習データが見つかりません ({pattern})")
            return
        
        latest_file = max(files, key=os.path.getctime)
        print(f"📖 データ読み込み: {latest_file}")
        df = pd.read_csv(latest_file)
        
        # 特徴量チェック
        valid_features = [c for c in self.predictor.feature_cols if c in df.columns]

        # 2. 最終モデル学習 (直近90%学習 - 10%確認)
        print(f"🛠 {self.timeframe}用モデルの構築 (直近重視)...")
        
        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        X_train = train_df[valid_features].fillna(0)
        y_train = train_df['label']
        X_val = val_df[valid_features].fillna(0)
        y_val = val_df['label']
        
        # LightGBM (分類) -> models/lgb_ETH_15m.pkl または _1h.pkl に保存
        self.predictor.train_lightgbm(X_train, y_train, X_val, y_val)

        # LightGBM (回帰)
        if 'future_change' in df.columns:
            y_reg_train = df.iloc[:split_idx]['future_change']
            y_reg_val = df.iloc[split_idx:]['future_change']
            self.predictor.train_regressor(X_train, y_reg_train, X_val, y_reg_val)
        
        # LSTM
        print(f"📊 LSTM学習 ({self.timeframe})...")
        prices = df['close'].values
        labels = df['label'].values
        self.predictor.train_lstm(prices, labels)
        
        print(f"✅ {self.timeframe} モデル学習完了")

if __name__ == "__main__":
    # 学習したい時間軸のリスト
    TARGET_TIMEFRAMES = ['15m', '1h']
    
    for tf in TARGET_TIMEFRAMES:
        try:
            trainer = ModelTrainer('ETH', timeframe=tf)
            trainer.train()
        except Exception as e:
            print(f"⚠️ {tf} の学習中にエラー発生: {e}")
            import traceback; traceback.print_exc()
            
    print("\n🎉 全モデルの学習プロセスが終了しました")