import pandas as pd
import glob
import os
from ml_predictor import MLPredictor

class ModelTrainer:
    def __init__(self, symbol='ETH'):
        self.symbol = symbol
        self.predictor = MLPredictor(symbol)
        
    def train(self):
        print("🚀 モデル学習開始 (時系列分割)")
        
        # 1. 最新データの特定
        files = glob.glob(f"training_data/{self.symbol}_15m_training*.csv")
        if not files:
            print("❌ エラー: 学習データが見つかりません。")
            # メッセージも修正しておくと親切です
            print("   先に 'python data_collector.py' を実行してデータを収集してください。")
            return
        
        latest_file = max(files, key=os.path.getctime)
        print(f"📖 データ読み込み: {latest_file}")
        df = pd.read_csv(latest_file)
        
        # 2. 特徴量の整合性チェック (重要)
        # MLPredictorが期待する特徴量がCSVに含まれているか確認
        expected_cols = self.predictor.feature_cols
        missing_cols = [c for c in expected_cols if c not in df.columns]
        
        if missing_cols:
            print(f"⚠️ 警告: CSVファイルに以下の特徴量が不足しています: {missing_cols}")
            print("   → 古いデータ形式の可能性があります。'python data_collector.py' を再実行してください。")
            # エラー回避のため、存在するカラムだけで学習を続行（またはここでreturnしても良い）
            valid_features = [c for c in expected_cols if c in df.columns]
        else:
            valid_features = expected_cols

        # 3. 時系列分割 (Train 80% / Val 20%)
        # 未来の情報をリークさせないため、シャッフルせずに前半・後半で分ける
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        # --- LightGBM 学習 ---
        print(f"📊 LightGBM学習: Train={len(train_df)}, Val={len(val_df)}")
        
        X_train = train_df[valid_features].fillna(0)
        y_train = train_df['label']
        X_val = val_df[valid_features].fillna(0)
        y_val = val_df['label']
        
        self.predictor.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # --- LSTM 学習 ---
        # LSTMは時系列シーケンスを作るため、全データを渡し内部で処理させる
        # (Kerasのvalidation_splitはデータの「後ろ」を使うため時系列的に整合する)
        print(f"📊 LSTM学習: 全データ数={len(df)}")
        
        prices = df['close'].values
        labels = df['label'].values
        
        self.predictor.train_lstm(prices, labels)
        
        print("✅ 全モデル学習完了")

if __name__ == "__main__":
    trainer = ModelTrainer('ETH')
    trainer.train()