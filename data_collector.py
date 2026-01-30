import numpy as np
import pandas as pd
from datetime import datetime
import os
import time
from advanced_market_data import AdvancedMarketData

class DataCollector:
    """
    修正版: Bot内部のオンライン学習用データ収集クラス
    - fetch_binance_data.py / ml_predictor.py と計算ロジックを完全統一
    - Botがバックグラウンドで自動実行します
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
        # Bot内部で呼び出されるログ
        # print(f"📥 [Auto] {timeframe}データ収集中... (目標: {limit}本)")
        
        # 1. 対象通貨のデータ取得
        df = self.market.get_ohlcv(timeframe=timeframe, limit=limit)
        if df is None or len(df) < 100:
            return None

        # 2. BTCデータの取得（相関特徴量用）
        df_btc = self.btc_market.get_ohlcv(timeframe=timeframe, limit=limit)
        
        # 3. データの結合とBTC特徴量の計算
        if df_btc is not None and len(df_btc) > 100:
            df = self.add_btc_features(df, df_btc)
        else:
            df['btc_correlation'] = 0.0
            df['btc_trend_strength'] = 0.0

        # 4. テクニカル指標計算 (ここが重要: 統一されたロジック)
        df = self.add_technical_indicators(df)
        
        # 5. その他のリアルタイム系特徴量（板情報などはAPIで取れないため0埋め）
        missing_features = ['orderbook_imbalance']
        for col in missing_features:
            df[col] = 0.0
        
        # ラベル作成
        df = self.create_labels(df, horizon=self.prediction_horizon)
        
        # 欠損値除去
        df = df.dropna()
        
        return df

    def add_btc_features(self, df: pd.DataFrame, df_btc: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(
            df, 
            df_btc[['timestamp', 'close', 'volume']], 
            on='timestamp', 
            how='inner', 
            suffixes=('', '_btc')
        )
        df = merged.copy()

        # BTC相関
        window_size = 24
        df['btc_correlation'] = df['close'].rolling(window=window_size).corr(df['close_btc']).fillna(0)

        # BTCトレンド強度
        btc_sma10 = df['close_btc'].rolling(10).mean()
        btc_sma30 = df['close_btc'].rolling(30).mean()
        df['btc_trend_strength'] = (btc_sma10 - btc_sma30) / btc_sma30 * 100
        df['btc_trend_strength'] = df['btc_trend_strength'].fillna(0)

        if 'close_btc' in df.columns: del df['close_btc']
        if 'volume_btc' in df.columns: del df['volume_btc']

        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        テクニカル指標と時間特徴量の追加 (ロジック統一版)
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
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        
        # --- 2. MACD ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal
        
        # --- 3. BB ---
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        df['bb_position'] = (close - (sma20 - 2*std20)) / (4*std20)
        df['bb_width'] = (4*std20) / sma20
        
        # --- 4. ATR ---
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
        
        # --- 7. 変動率 & Lag (★追加: ml_predictor.pyと統一) ---
        current_return = close.pct_change(1).fillna(0) * 100
        df['price_change_1h'] = current_return
        df['price_change_4h'] = close.pct_change(4).fillna(0) * 100
        
        df['return_lag_1'] = current_return.shift(1).fillna(0)
        df['return_lag_2'] = current_return.shift(2).fillna(0)
        df['return_lag_3'] = current_return.shift(3).fillna(0)
        
        # --- 8. Volatility & Ratio (★追加: ml_predictor.pyと統一) ---
        df['volatility'] = close.rolling(20).std() / sma20 * 100
        
        long_term_atr = df['atr'].rolling(10).mean().replace(0, 1)
        df['volatility_ratio'] = df['atr'] / long_term_atr
        
        # --- 9. 時間特徴量 ---
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
            df['hour_sin'] = np.sin(2 * np.pi * dates.dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * dates.dt.hour / 24)
            df['day_of_week'] = dates.dt.dayofweek / 6.0
        else:
            df['hour_sin'] = 0; df['hour_cos'] = 0; df['day_of_week'] = 0

        return df

    def create_labels(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        ラベルと回帰ターゲットの作成
        """
        future_price = df['close'].shift(-horizon)
        current_price = df['close']
        
        # 回帰用ターゲット (これがRegモデルの学習に必須)
        pct_change = ((future_price - current_price) / current_price) * 100
        df['future_change'] = pct_change
        
        if 'atr' in df.columns:
            atr_pct = (df['atr'] / df['close']) * 100
            dynamic_threshold = (atr_pct * 0.20).clip(0.08, 1.2)
        else:
            dynamic_threshold = pd.Series(self.neutral_threshold, index=df.index)

        conditions = [
            (pct_change > dynamic_threshold),
            (pct_change < -dynamic_threshold)
        ]
        choices = [1, -1]
        
        df['label'] = np.select(conditions, choices, default=0)
        
        return df
    
    # 互換性のためのダミーメソッド
    def save_dataset(self, df, filename): pass 
    def collect_multiple_timeframes(self): pass