import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
from advanced_market_data import AdvancedMarketData

class DataCollector:
    """
    修正版: 3値分類（上昇/下降/中立）データ収集
    - Pandasベクトル演算により高速に学習データを生成
    - ATRベースの動的ラベル付けを実装
    - ✅ BTC相関・トレンド強度を実データから計算して付与
    - ℹ️ 板情報(orderbook)は過去データ取得不可のため0埋め継続
    """
    
    def __init__(self, symbol='ETH', data_dir='training_data'):
        self.symbol = symbol
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # 対象通貨のマーケットデータ
        self.market = AdvancedMarketData(symbol)
        # BTC相関算出用のマーケットデータ
        self.btc_market = AdvancedMarketData('BTC')
        
        # デイトレ用にホライゾンを短縮 (1本先の予測)
        self.prediction_horizon = 1 
        # 変動率閾値のベース (ATRがない場合のフォールバック)
        self.neutral_threshold = 0.3 
    
    def collect_historical_data(self, timeframe='1h', limit=2000):
        print(f"\n📥 {timeframe}足データ収集中... (目標: {limit}本)")
        
        # 1. 対象通貨のデータ取得
        df = self.market.get_ohlcv(timeframe=timeframe, limit=limit)
        if df is None or len(df) < 100:
            print("⚠️ データ不足または取得失敗")
            return None

        # 2. BTCデータの取得（相関特徴量用）
        print(f"   ➕ BTCデータ同期中...")
        df_btc = self.btc_market.get_ohlcv(timeframe=timeframe, limit=limit)
        
        # 3. データの結合とBTC特徴量の計算
        if df_btc is not None and len(df_btc) > 100:
            df = self.add_btc_features(df, df_btc)
        else:
            print("⚠️ BTCデータ取得失敗のため、BTC特徴量は0で埋めます")
            df['btc_correlation'] = 0.0
            df['btc_trend_strength'] = 0.0

        # 4. テクニカル指標計算 (Seriesとして一括計算)
        df = self.add_technical_indicators(df)
        
        # 5. その他のリアルタイム系特徴量（板情報など）
        # ※過去の板情報はAPIで取得できないため、引き続き0.0で初期化します
        #   (ライブトレード時には advanced_market_data.py がリアルタイム値を取得します)
        missing_features = ['orderbook_imbalance']
        for col in missing_features:
            df[col] = 0.0
        
        # ラベル作成 (ATR動的閾値)
        df = self.create_labels(df, horizon=self.prediction_horizon)
        
        # 欠損値除去 (SMA計算や相関計算で発生したNaNを消す)
        df = df.dropna()
        
        return df

    def add_btc_features(self, df: pd.DataFrame, df_btc: pd.DataFrame) -> pd.DataFrame:
        """
        BTCデータとマージして相関とトレンド強度を計算
        """
        # タイムスタンプでマージ (inner joinで両方存在する期間のみ残す)
        # suffixesを使ってカラム名を区別: close -> close_target, close_btc
        merged = pd.merge(
            df, 
            df_btc[['timestamp', 'close', 'volume']], 
            on='timestamp', 
            how='inner', 
            suffixes=('', '_btc')
        )
        
        # マージによって行が減る可能性があるため、dfを更新
        df = merged.copy()

        # --- 1. BTC相関 (Rolling Correlation) ---
        # 過去24期間（例えば1時間足なら24時間）の相関
        window_size = 24
        df['btc_correlation'] = df['close'].rolling(window=window_size).corr(df['close_btc'])
        
        # NaN埋め (相関なし=0)
        df['btc_correlation'] = df['btc_correlation'].fillna(0)

        # --- 2. BTCトレンド強度 ---
        # 短期(10)と長期(30)の移動平均の乖離率をトレンド強度とする
        btc_sma10 = df['close_btc'].rolling(10).mean()
        btc_sma30 = df['close_btc'].rolling(30).mean()
        
        # トレンド強度: (短期 - 長期) / 長期 * 100
        # プラスなら上昇トレンド、マイナスなら下降トレンドの強さ
        df['btc_trend_strength'] = (btc_sma10 - btc_sma30) / btc_sma30 * 100
        df['btc_trend_strength'] = df['btc_trend_strength'].fillna(0)

        # 不要なBTCカラムを削除 (close_btc, volume_btc)
        # メモリ節約のため
        if 'close_btc' in df.columns:
            del df['close_btc']
        if 'volume_btc' in df.columns:
            del df['volume_btc']

        print(f"   ✅ BTC特徴量計算完了 (相関 & トレンド強度)")
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        テクニカル指標と時間特徴量の追加
        """
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # --- 1. RSI (14) ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # --- 2. MACD (12, 26, 9) ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal
        
        # --- 3. Bollinger Bands (20, 2) ---
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        df['bb_position'] = (close - (sma20 - 2*std20)) / (4*std20)
        df['bb_width'] = (4*std20) / sma20
        
        # --- 4. ATR (14) ---
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        # --- 5. SMA & Ratio ---
        df['sma_20'] = sma20
        df['sma_50'] = close.rolling(50).mean()
        df['sma_20_50_ratio'] = (df['sma_20'] / df['sma_50'] - 1) * 100
        
        # --- 6. Volume Ratio ---
        vol_ma = volume.rolling(20).mean()
        df['volume_ratio'] = volume / vol_ma.replace(0, 1)
        
        # --- 7. 変動率 ---
        df['price_change_1h'] = close.pct_change(1) * 100
        df['price_change_4h'] = close.pct_change(4) * 100
        
        # --- 8. ボラティリティ ---
        df['volatility'] = close.rolling(20).std() / sma20 * 100
        
        # --- 9. 時間特徴量 ---
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
            df['hour_sin'] = np.sin(2 * np.pi * dates.dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * dates.dt.hour / 24)
            df['day_of_week'] = dates.dt.dayofweek / 6.0
        else:
            df['hour_sin'] = 0
            df['hour_cos'] = 0
            df['day_of_week'] = 0

        return df

    def create_labels(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        ATRに基づいた動的閾値によるラベル付け
        """
        future_price = df['close'].shift(-horizon)
        current_price = df['close']
        
        pct_change = ((future_price - current_price) / current_price) * 100
        df['future_change'] = pct_change
        
        if 'atr' in df.columns:
            atr_pct = (df['atr'] / df['close']) * 100
            dynamic_threshold = (atr_pct * 0.35).clip(0.1, 1.5)
        else:
            dynamic_threshold = pd.Series(self.neutral_threshold, index=df.index)

        conditions = [
            (pct_change > dynamic_threshold),
            (pct_change < -dynamic_threshold)
        ]
        choices = [1, -1]
        
        df['label'] = np.select(conditions, choices, default=0)
        
        return df

    def save_dataset(self, df: pd.DataFrame, filename: str = None):
        if filename is None:
            filename = f"{self.symbol}_training.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 データ保存完了: {filepath} ({len(df)}行)")
        
        counts = df['label'].value_counts().sort_index()
        dist = counts.to_dict()
        print(f"   分布: {dist}")
        return filepath
    
    def collect_multiple_timeframes(self):
        # デイトレ用 15分足
        filename = f"{self.symbol}_15m_training.csv"
        
        df_15m = self.collect_historical_data('15m', 3000)
        if df_15m is not None:
            path = self.save_dataset(df_15m, filename=filename)
            return {'15m': path}
        else:
            print("❌ 15mデータの取得に失敗しました")
            return {}

if __name__ == "__main__":
    c = DataCollector('ETH')
    c.collect_multiple_timeframes()