import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from ml_predictor import MLPredictor

class ModelTrainer:
    def __init__(self, symbol='ETH'):
        self.symbol = symbol
        self.predictor = MLPredictor(symbol)
        
    def train(self):
        print("🚀 モデル学習開始 (ウォークフォワード検証 & 最終学習)")
        
        # 1. 最新データの特定
        files = glob.glob(f"training_data/{self.symbol}_15m_training*.csv")
        if not files:
            print("❌ エラー: 学習データが見つかりません。")
            print("   先に 'python data_collector.py' を実行してデータを収集してください。")
            return
        
        latest_file = max(files, key=os.path.getctime)
        print(f"📖 データ読み込み: {latest_file}")
        df = pd.read_csv(latest_file)
        
        # 2. 特徴量の整合性チェック
        expected_cols = self.predictor.feature_cols
        missing_cols = [c for c in expected_cols if c not in df.columns]
        
        if missing_cols:
            print(f"⚠️ 警告: CSVファイルに以下の特徴量が不足しています: {missing_cols}")
            print("   → 'python data_collector.py' を再実行することを推奨します。")
            valid_features = [c for c in expected_cols if c in df.columns]
        else:
            valid_features = expected_cols

        # ---------------------------------------------------------
        # ウォークフォワード検証 (Walk-Forward Validation)
        # ---------------------------------------------------------
        print("\n🔍 ウォークフォワード検証を開始 (5分割)...")
        print("   ※過去のデータだけに過剰適合していないかチェックします")

        tscv = TimeSeriesSplit(n_splits=5)
        X = df[valid_features].fillna(0)
        y = df['label']
        
        scores = []
        fold = 1
        
        # データをずらしながら検証を繰り返す
        for train_index, val_index in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            
            # 検証用に一時的に学習 (注: ここでは精度確認が目的)
            self.predictor.train_lightgbm(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # 精度を評価
            score = self.predictor.evaluate_model(self.predictor.lgb_model, X_val_fold, y_val_fold)
            scores.append(score)
            print(f"   [Fold {fold}] Train:{len(X_train_fold)} -> Val:{len(X_val_fold)} | Accuracy: {score:.4f}")
            fold += 1

        avg_score = np.mean(scores)
        print(f"\n📊 検証スコア平均: {avg_score:.4f}")
        if avg_score < 0.4:
            print("⚠️ 注意: モデルの予測精度が低めです。特徴量の見直しが必要かもしれません。")
        else:
            print("✅ 安定した精度が出ています。カーブフィッティングの可能性は低いです。")

        # ---------------------------------------------------------
        # 4. 本番用最終モデルの学習 (直近データを重視)
        # ---------------------------------------------------------
        print("\n🛠 本番用最終モデルの構築 (全データの直近90%学習 - 10%確認)")
        
        # 直近の市場環境に合わせるため、最後の10%を検証に残して学習
        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        X_train = train_df[valid_features].fillna(0)
        y_train = train_df['label']
        X_val = val_df[valid_features].fillna(0)
        y_val = val_df['label']
        
        # LightGBM 保存用学習
        self.predictor.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # LSTM 学習 (全期間のシーケンスを使用)
        print(f"📊 LSTM学習: 全データ数={len(df)}")
        prices = df['close'].values
        labels = df['label'].values
        self.predictor.train_lstm(prices, labels)
        
        print("✅ 全モデル学習完了")

if __name__ == "__main__":
    trainer = ModelTrainer('ETH')
    trainer.train()