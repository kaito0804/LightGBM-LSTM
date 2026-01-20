# simple_backtest.py
import pandas as pd
import numpy as np
import os
from advanced_market_data import AdvancedMarketData

def run_backtest(symbol='ETH', days=365):
    print(f"🧪 バックテスト開始: {symbol} 過去{days}日間")
    
    # データ取得（AdvancedMarketDataを利用）
    market = AdvancedMarketData(symbol)
    # 1時間足を取得
    limit = 24 * days
    df = market.get_ohlcv('1h', limit=limit)
    
    if df is None or df.empty:
        print("データ取得失敗")
        return

    # テクニカル指標計算（簡易的）
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = market.calculate_rsi(df['close'].values)

    # シミュレーション設定
    initial_capital = 1000.0
    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    fee_rate = 0.00035 * 2  # 往復手数料 (Taker想定)
    trades = []
    
    print(f"   データ数: {len(df)}本")
    print("   シミュレーション実行中...")

    # ループ処理
    for i in range(50, len(df)):
        row = df.iloc[i]
        price = row['close']
        
        # --- 簡易ロジック (本来はMLモデルの予測を使う) ---
        # ゴールデンクロス ＆ RSI売られすぎ
        buy_signal = (row['sma_20'] > row['sma_50']) and (row['rsi'] < 40)
        
        # 利益確定(1%) または 損切り(-3%) または デッドクロス
        sell_signal = False
        if position > 0:
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct >= 0.01: # 利確
                sell_signal = True
            elif pnl_pct <= -0.03: # 損切り
                sell_signal = True
            elif row['sma_20'] < row['sma_50']: # トレンド転換
                sell_signal = True

        # 取引実行
        if position == 0 and buy_signal:
            # 全力買い（レバ1倍）
            position = (capital * 0.99) / price
            entry_price = price
            capital -= position * price * (1 + 0.00035) # エントリー手数料
            trades.append({'type': 'BUY', 'price': price, 'time': row['timestamp']})
            
        elif position > 0 and sell_signal:
            # 決済
            revenue = position * price
            fee = revenue * 0.00035
            capital += (revenue - fee)
            
            pnl = (price - entry_price) / entry_price * 100
            trades.append({'type': 'SELL', 'price': price, 'pnl': pnl, 'time': row['timestamp']})
            position = 0.0
            entry_price = 0.0

    # 最終評価
    if position > 0:
        capital += position * df.iloc[-1]['close']

    total_return = (capital - initial_capital) / initial_capital * 100
    
    print("\n" + "="*50)
    print(f"📊 バックテスト結果")
    print(f"   初期資金: ${initial_capital:.2f}")
    print(f"   最終資金: ${capital:.2f}")
    print(f"   総収益率: {total_return:.2f}%")
    print(f"   取引回数: {len(trades)}")
    print(f"   勝率: {len([t for t in trades if t.get('pnl',0)>0]) / (len(trades)/2)*100:.1f}%" if len(trades)>0 else "   勝率: N/A")
    print("="*50)

if __name__ == "__main__":
    run_backtest()