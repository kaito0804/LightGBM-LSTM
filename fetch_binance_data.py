# fetch_binance_data.py (統合版)
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os

# === 設定エリア ===
SYMBOL_TARGET = 'ETHUSDT'
SYMBOL_BTC = 'BTCUSDT'
OUTPUT_DIR = 'training_data'

# 取得する時間軸と期間のリスト
FETCH_CONFIGS = [
    {'timeframe': '15m', 'days': 100}, # 短期は直近重視
    {'timeframe': '1h',  'days': 365}  # 長期は1年分確保
]

def fetch_binance_klines(symbol, interval, days):
    """Binanceからローソク足を取得"""
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    
    end_time = int(time.time() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    print(f"   📥 {symbol} ({interval}) 取得中... ({days}日分)")
    
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
            if not isinstance(data, list) or len(data) == 0: break
            
            all_klines.extend(data)
            current_start = data[-1][0] + 1
            if current_start >= end_time: break
            time.sleep(0.05) # 少しウェイト
            
        except Exception as e:
            print(f"⚠️ エラー: {e}")
            break
            
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

def calculate_features(df, df_btc):
    """共通特徴量計算ロジック"""
    # BTCデータのマージ
    df = pd.merge_asof(
        df.sort_values('timestamp'),
        df_btc[['timestamp', 'close']].sort_values('timestamp').rename(columns={'close': 'close_btc'}),
        on='timestamp',
        direction='nearest'
    )
    
    close = df['close']
    volume = df['volume']
    
    # テクニカル指標
    df['btc_correlation'] = close.rolling(24).corr(df['close_btc']).fillna(0)
    
    btc_sma10 = df['close_btc'].rolling(10).mean()
    btc_sma30 = df['close_btc'].rolling(30).mean()
    df['btc_trend_strength'] = ((btc_sma10 - btc_sma30) / btc_sma30 * 100).fillna(0)
    
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
    high, low = df['high'], df['low']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # SMA Ratio & Volume
    df['sma_20'] = sma20
    df['sma_50'] = close.rolling(50).mean()
    df['sma_20_50_ratio'] = (df['sma_20'] / df['sma_50'] - 1) * 100
    
    vol_ma = volume.rolling(20).mean()
    df['volume_ratio'] = volume / vol_ma.replace(0, 1)
    
    # Changes & Lags
    current_return = close.pct_change(1).fillna(0) * 100
    df['price_change_1h'] = current_return
    df['price_change_4h'] = close.pct_change(4).fillna(0) * 100
    
    df['return_lag_1'] = current_return.shift(1).fillna(0)
    df['return_lag_2'] = current_return.shift(2).fillna(0)
    df['return_lag_3'] = current_return.shift(3).fillna(0)
    
    # Volatility Ratio
    long_term_atr = df['atr'].rolling(10).mean().replace(0, 1)
    df['volatility_ratio'] = df['atr'] / long_term_atr
    df['volatility'] = close.rolling(20).std() / sma20 * 100
    
    # Time
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['day_of_week'] = df['timestamp'].dt.dayofweek / 6.0
    df['orderbook_imbalance'] = 0.0

    # Labels
    horizon = 1
    future_change = close.shift(-horizon).pct_change(1) * 100
    df['future_change'] = (df['close'].shift(-horizon) - df['close']) / df['close'] * 100
    
    atr_pct = (df['atr'] / close) * 100
    threshold = (atr_pct * 0.20).clip(0.08, 1.2)
    
    conditions = [(df['future_change'] > threshold), (df['future_change'] < -threshold)]
    choices = [1, -1]
    df['label'] = np.select(conditions, choices, default=0)
    
    return df.dropna()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 データ取得開始: {len(FETCH_CONFIGS)}つの設定")

    for config in FETCH_CONFIGS:
        tf = config['timeframe']
        days = config['days']
        output_file = f"{OUTPUT_DIR}/ETH_{tf}_training.csv"
        
        print(f"\n[{tf} データ取得] 開始...")
        df_eth = fetch_binance_klines(SYMBOL_TARGET, tf, days)
        df_btc = fetch_binance_klines(SYMBOL_BTC, tf, days)
        
        if len(df_eth) > 0 and len(df_btc) > 0:
            df_final = calculate_features(df_eth, df_btc)
            df_final.to_csv(output_file, index=False)
            print(f"✅ 完了: {output_file} ({len(df_final)}行)")
        else:
            print(f"❌ 失敗: {tf} のデータが取れませんでした")

    print("\n🎉 全データ取得プロセス完了！")

if __name__ == "__main__":
    main()