# fetch_binance_data.py
# Binanceから過去データを大量に取得し、Hyperliquid Bot用の学習データを作成するツール

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os

# 設定
SYMBOL_TARGET = 'ETHUSDT' # ターゲット通貨
SYMBOL_BTC = 'BTCUSDT'    # 相関用BTC
TIMEFRAME = '15m'         # 足（Botの設定に合わせる）
DAYS_TO_FETCH = 100       # 取得する期間（1年分 = 約35,000本）
OUTPUT_DIR = 'training_data'

# 保存ファイル名（Botが読み込むファイル名に合わせる）
OUTPUT_FILENAME = f"{OUTPUT_DIR}/ETH_{TIMEFRAME}_training.csv"

def fetch_binance_klines(symbol, interval, days):
    """Binanceからローソク足を取得する"""
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    
    # 開始時刻の計算 (ミリ秒)
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    print(f"📥 {symbol} のデータをBinanceから取得中 ({days}日分)...")
    
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'startTime': current_start,
            'endTime': end_time
        }
        
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not isinstance(data, list) or len(data) == 0:
                break
            
            all_klines.extend(data)
            
            # 次の取得開始時刻を設定（最後のローソク足の時刻 + 1ms）
            last_timestamp = data[-1][0]
            current_start = last_timestamp + 1
            
            # 進捗表示
            fetched_date = datetime.fromtimestamp(last_timestamp / 1000)
            print(f"   ... {fetched_date.strftime('%Y-%m-%d')} まで取得 ({len(all_klines)}本)")
            
            if current_start >= end_time:
                break
                
            # API制限考慮
            time.sleep(0.1)
            
        except Exception as e:
            print(f"⚠️ エラー発生: {e}")
            break
            
    # DataFrame化
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # 型変換
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 必要な列のみ残す
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def calculate_features(df, df_btc):
    """
    Botと同じロジックで特徴量を計算する (整合性確保版)
    """
    print("🛠 特徴量エンジニアリング中 (Lag/Volatility追加)...")
    
    # 1. BTCデータのマージ（相関計算用）
    df = pd.merge_asof(
        df.sort_values('timestamp'),
        df_btc[['timestamp', 'close']].sort_values('timestamp').rename(columns={'close': 'close_btc'}),
        on='timestamp',
        direction='nearest'
    )
    
    # --- テクニカル指標（Botのロジックを再現）---
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # BTC相関
    df['btc_correlation'] = close.rolling(24).corr(df['close_btc']).fillna(0)
    
    # BTCトレンド強度
    btc_sma10 = df['close_btc'].rolling(10).mean()
    btc_sma30 = df['close_btc'].rolling(30).mean()
    df['btc_trend_strength'] = ((btc_sma10 - btc_sma30) / btc_sma30 * 100).fillna(0)
    
    # RSI (14)
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
    
    # ATR (整合性のため計算式を統一)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # SMA & Volume Ratio
    df['sma_20'] = sma20
    df['sma_50'] = close.rolling(50).mean()
    df['sma_20_50_ratio'] = (df['sma_20'] / df['sma_50'] - 1) * 100
    
    vol_ma = volume.rolling(20).mean()
    df['volume_ratio'] = volume / vol_ma.replace(0, 1)
    
    # --- ★ここから修正・追加箇所 ---
    # ml_predictor.py と整合性を取るため、'price_change_1h' は「1本前(15m)の変化率」とする
    current_return = close.pct_change(1).fillna(0) * 100
    df['price_change_1h'] = current_return
    
    # 4本前(本来の1h)の変化率も特徴量として残す
    df['price_change_4h'] = close.pct_change(4).fillna(0) * 100 
    
    # ★Lag特徴量 (直近の勢い)
    df['return_lag_1'] = current_return.shift(1).fillna(0)
    df['return_lag_2'] = current_return.shift(2).fillna(0)
    df['return_lag_3'] = current_return.shift(3).fillna(0)
    
    # ★Volatility Ratio (ボラティリティの拡大度)
    long_term_atr = df['atr'].rolling(10).mean().replace(0, 1)
    df['volatility_ratio'] = df['atr'] / long_term_atr
    
    df['volatility'] = close.rolling(20).std() / sma20 * 100
    # --------------------------------
    
    # 時間特徴量
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['day_of_week'] = df['timestamp'].dt.dayofweek / 6.0
    
    # 板情報はBinance過去データにないので0埋め
    df['orderbook_imbalance'] = 0.0

    # --- ラベル作成 (正解データ) ---
    horizon = 1 # 1本先
    future_change = close.shift(-horizon).pct_change(1) * 100 # 次の足の変化率
    df['future_change'] = (df['close'].shift(-horizon) - df['close']) / df['close'] * 100
    
    # ATRベースの動的閾値でラベル付け
    atr_pct = (df['atr'] / close) * 100
    threshold = (atr_pct * 0.20).clip(0.08, 1.2)
    
    conditions = [
        (df['future_change'] > threshold),
        (df['future_change'] < -threshold)
    ]
    choices = [1, -1] # Buy, Sell
    df['label'] = np.select(conditions, choices, default=0) # Hold
    
    return df.dropna()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. データの取得
    df_eth = fetch_binance_klines(SYMBOL_TARGET, TIMEFRAME, DAYS_TO_FETCH)
    df_btc = fetch_binance_klines(SYMBOL_BTC, TIMEFRAME, DAYS_TO_FETCH)
    
    if len(df_eth) == 0 or len(df_btc) == 0:
        print("❌ データ取得に失敗しました")
        return

    # 2. マージと特徴量計算
    df_final = calculate_features(df_eth, df_btc)
    
    # 3. 保存
    df_final.to_csv(OUTPUT_FILENAME, index=False)
    
    print("\n" + "="*50)
    print(f"✅ 学習データ作成完了 (修正版)！")
    print(f"📁 保存先: {OUTPUT_FILENAME}")
    print(f"📊 データ数: {len(df_final)} 行 (約{DAYS_TO_FETCH}日分)")
    print(f"📈 ラベル分布: {df_final['label'].value_counts().to_dict()}")
    print("="*50)
    print("\n👉 次のステップ: 'python train_models.py' を実行してAIを再学習させてください")

if __name__ == "__main__":
    main()