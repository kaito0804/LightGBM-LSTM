# advanced_market_data.py (マルチタイムフレーム対応修正版)
# 高度な市場データ取得とテクニカル分析（Mainnet対応版）

import requests
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()
from improved_signal_scoring import ImprovedSignalScoring


class AdvancedMarketData:
    """
    高度な市場データ分析クラス（Mainnet対応版）
    - Hyperliquid APIから実際のローソク足データを取得
    - Mainnetではフォールバックデータを使用せず、エラーで停止
    """
    
    VALID_INTERVALS = {'1m', '5m', '15m', '1h', '4h', '1d'}

    def __init__(self, symbol='ETH'):
        # ✅ シンボル名を正規化 (より堅牢に)
        self.symbol = symbol.replace('-USD', '').replace('/USD', '').upper()

        # 改善版スコアリングシステムの初期化
        self.scorer = ImprovedSignalScoring()
        
        # ネットワーク設定
        self.network = os.getenv("NETWORK", "testnet").lower()
        
        # Hyperliquid API設定
        self.api_base = "https://api.hyperliquid.xyz"
        
        self.info_url = f"{self.api_base}/info"

        # デイトレードの主軸を環境変数から取得（なければ15m）
        self.main_timeframe = os.getenv("MAIN_TIMEFRAME", "15m")
        
        print(f"📊 AdvancedMarketData初期化")
        print(f"   ネットワーク: {self.network.upper()}")
        print(f"   シンボル: {self.symbol}")
        print(f"   API: {self.api_base}")
        print(f"   主軸タイムフレーム: {self.main_timeframe}") 
        
        if self.network == "mainnet":
            print(f"   ⚠️ Mainnetモード: フォールバックデータ無効")
    


    def _get_interval_string(self, timeframe: str) -> str:
        """
        辞書マッピングを廃止し、セットによる検証に変更
        """
        if timeframe in self.VALID_INTERVALS:
            return timeframe
        
        # 不正な値が来た場合はデフォルト '1h' を返しつつ警告
        print(f"⚠️ 無効なタイムフレーム '{timeframe}'。デフォルトの '1h' を使用します。")
        return '1h'
    


    def get_ohlcv(self, timeframe='1h', limit=500):
        """
        OHLCV（ローソク足）データ取得（実データ）
        Mainnetではエラー時に停止、Testnetではフォールバック可
        """
        try:
            interval_str = self._get_interval_string(timeframe)
            
            # 時間計算
            interval_ms_map = {
                '1m': 60000, '5m': 300000, '15m': 900000,
                '1h': 3600000, '4h': 14400000, '1d': 86400000
            }
            duration_ms = limit * interval_ms_map.get(timeframe, 3600000)
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - duration_ms
            
            # APIリクエスト
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": self.symbol,
                    "interval": interval_str,
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
            
            response = requests.post(self.info_url, json=payload, timeout=10)
            
            # ステータスコードチェック
            if response.status_code != 200:
                raise ValueError(f"API応答エラー: {response.status_code}")
            
            data = response.json()
            if not data:
                raise ValueError(f"データが空です: {self.symbol} {timeframe}")
            
            # データパース (辞書形式を想定)
            candles = []
            for c in data:
                if isinstance(c, dict):
                    candles.append({
                        'timestamp': pd.to_datetime(c['t'], unit='ms'),
                        'open': float(c['o']),
                        'high': float(c['h']),
                        'low': float(c['l']),
                        'close': float(c['c']),
                        'volume': float(c.get('v', 0))
                    })
            
            if not candles:
                raise ValueError("ローソク足データのパースに失敗")
            
            # DataFrame化
            df = pd.DataFrame(candles)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            # エラー処理の一元化
            error_msg = f"OHLCV取得エラー: {str(e)}"
            print(f"⚠️ {error_msg}")
            
            return None
    


    def _get_fallback_data(self, limit):
        # (省略: このメソッドは使用されません)
        return None


    
    def calculate_rsi(self, prices, period=14):
        """
        RSI計算 - Wilder's平滑化方式
        """
        if len(prices) < period + 2:
            return 50.0
        
        prices_series = pd.Series(prices)
        deltas = prices_series.diff()
        
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        curr_gain = avg_gain.iloc[-1]
        curr_loss = avg_loss.iloc[-1]
        
        if curr_loss < 1e-10:
            return 100.0 if curr_gain > 0 else 50.0
        
        rs = curr_gain / curr_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)



    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        MACD計算（Pandas標準）
        """
        min_required = slow + signal + 10
        if len(prices) < min_required:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
        ema_slow = prices_series.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        def safe_get(series):
            val = series.iloc[-1]
            return 0.0 if pd.isna(val) else float(val)

        return {
            'macd': safe_get(macd),
            'signal': safe_get(signal_line),
            'histogram': safe_get(histogram)
        }



    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        ボリンジャーバンド計算
        """
        if len(prices) < period:
            current = float(prices[-1]) if len(prices) > 0 else 0.0
            return {
                'upper': current, 'middle': current, 'lower': current,
                'position': 0.5, 'width': 0.0
            }
        
        prices_series = pd.Series(prices)
        sma = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std(ddof=0)
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        def get_val(series):
            val = series.iloc[-1]
            return 0.0 if pd.isna(val) else float(val)

        upper_val = get_val(upper)
        lower_val = get_val(lower)
        sma_val = get_val(sma)
        current_price = float(prices[-1])
        
        if upper_val > lower_val:
            position = (current_price - lower_val) / (upper_val - lower_val)
        else:
            position = 0.5
            
        if sma_val != 0:
            width = (upper_val - lower_val) / sma_val
        else:
            width = 0.0
        
        return {
            'upper': upper_val, 'middle': sma_val, 'lower': lower_val,
            'position': float(position), 'width': float(width)
        }



    def calculate_atr(self, df, period=14):
        """
        ATR計算 - Wilder's平滑化方式
        """
        if len(df) < period:
            if len(df) > 0: return float((df['high'] - df['low']).mean())
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        val = atr.iloc[-1]
        return 0.0 if pd.isna(val) else float(val)



    def get_comprehensive_analysis(self, interval=None):
        """
        総合的な市場分析（改善版スコアリング統合）
        ✅ 修正: 引数 interval を受け取り、指定された時間軸のデータをメインにセットする
        """
        analysis = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframes': {},
            'indicators': {}, 
            'trend': {},
            'signal_strength': 0,
            'recommendation': 'HOLD',
            'volatility': 0.0,
            'sentiment': 'NEUTRAL',
            'market_structure': {'orderbook_imbalance': 0.0, 'btc_trend': 0.0}
        }
        
        scoring_data = {}
        
        # タイムフレームごとの取得設定
        timeframe_config = {
            '15m': {'limit': 300}, 
            '1h': {'limit': 400},  
            '4h': {'limit': 500}   
        }

        # メインターゲットの決定 (引数があればそれを、なければ環境変数の値)
        target_tf = interval if interval else self.main_timeframe

        # --- 1. データ収集とテクニカル計算 ---
        for tf, config in timeframe_config.items():
            df = self.get_ohlcv(timeframe=tf, limit=config['limit'])
            
            if df is None or len(df) < 50:
                print(f"⚠️ {tf} データ不足のため分析スキップ")
                continue
            
            prices = df['close'].values
            
            rsi = self.calculate_rsi(prices)
            macd = self.calculate_macd(prices)
            bb = self.calculate_bollinger_bands(prices)
            atr = self.calculate_atr(df)
            
            sma_20 = float(np.mean(prices[-20:]))
            sma_50 = float(np.mean(prices[-50:]))
            sma_200 = float(np.mean(prices[-200:])) if len(prices) >= 200 else None
            
            trend_dir = "上昇" if sma_20 > sma_50 else "下降"
            trend_str = abs(sma_20 - sma_50) / sma_50 * 100 if sma_50 != 0 else 0
            
            vol_period = min(20, len(prices))
            volatility = float(np.std(prices[-vol_period:]) / np.mean(prices[-vol_period:]) * 100)

            if volatility > self.scorer.extreme_vol_threshold:
                print(f"⚠️ {tf} ボラティリティ過大: {volatility:.2f}%")
            
            scoring_data[tf] = {
                'rsi': rsi,
                'macd': macd,
                'bb': bb,
                'prices': prices,
                'volatility': volatility,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200
            }
            
            tf_data = {
                'current_price': float(prices[-1]),
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bb,
                'atr': atr,
                'trend': trend_dir,
                'trend_strength': trend_str,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'volatility': volatility,
                'volume': float(df['volume'].iloc[-1]),
                'price_change_24h': float(((prices[-1] - prices[0]) / prices[0]) * 100),
                'prices': prices,
                'df_summary': df.iloc[-1].to_dict()
            }
            analysis['timeframes'][tf] = tf_data
            
            # --- ★修正箇所: 指定されたターゲット時間軸のデータを優先採用 ---
            if tf == target_tf:
                analysis['indicators'] = {
                    'rsi': rsi,
                    'macd': macd,
                    'bollinger': bb,
                    'atr': atr
                }
                analysis['volatility'] = volatility # トップレベルのvolatilityも更新
                analysis['trend'] = {'direction': trend_dir, 'strength': trend_str}

        # --- 2. 総合スコアリング ---
        if scoring_data:
            scoring_result = self.scorer.calculate_comprehensive_score(scoring_data)
            
            analysis['signal_strength'] = scoring_result['signal_strength']
            # analysis['volatility'] は上でターゲット時間軸のものに設定済みなので上書きしない
            # もしスコアリング全体のvolatilityを使いたい場合は以下を生かす
            # analysis['volatility'] = scoring_result['volatility']
            
            analysis['sentiment'] = scoring_result['direction']
            
            strength = scoring_result['signal_strength']
            direction = scoring_result['direction']
            
            if strength > 70:
                rec = 'STRONG_BUY' if direction == 'BULLISH' else 'STRONG_SELL'
            elif strength > 55:
                rec = 'BUY' if direction == 'BULLISH' else 'SELL'
            elif strength < 30: 
                rec = 'STRONG_SELL' if direction == 'BEARISH' else 'STRONG_BUY' 
            elif strength < 45:
                rec = 'SELL' if direction == 'BEARISH' else 'BUY'
            else:
                rec = 'HOLD'
            
            analysis['recommendation'] = rec
            analysis['market_regime'] = scoring_result['regime']
            analysis['trend_strength'] = scoring_result['trend_strength']
            analysis['scoring_breakdown'] = scoring_result.get('breakdown', {})
        
        # --- 3. 市場構造データの統合 ---
        try:
            structure = self.get_market_structure_features()
            if structure:
                analysis['market_structure'] = structure
        except Exception as e:
            print(f"⚠️ 市場構造データの統合に失敗: {e}")
        
        return analysis



    def get_current_price(self):
        """
        現在価格を取得 (allMids)
        """
        try:
            payload = {"type": "allMids"}
            response = requests.post(self.info_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if self.symbol in data:
                    return float(data[self.symbol])
            
            raise ValueError(f"allMids取得失敗: Status {response.status_code}")

        except Exception as e:
            print(f"⚠️ 現在価格取得エラー: {e}")
            return None



    def get_market_structure_features(self):
        """
        AI用の追加特徴量を取得（板の偏り、BTC相関）
        """
        features = {
            'orderbook_imbalance': 0.0,
            'btc_trend': 0.0
        }
        
        try:
            # 1. 板情報の不均衡
            payload = {"type": "l2Snapshot", "coin": self.symbol}
            response = requests.post(self.info_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'levels' in data and len(data['levels']) >= 2:
                    bids = data['levels'][0]
                    asks = data['levels'][1]
                    
                    if bids and asks:
                        bid_vol = sum([float(x['sz']) for x in bids[:10]])
                        ask_vol = sum([float(x['sz']) for x in asks[:10]])
                        total_vol = bid_vol + ask_vol
                        if total_vol > 0:
                            features['orderbook_imbalance'] = (bid_vol - ask_vol) / total_vol
            
            # 2. BTCトレンド
            if self.symbol != 'BTC':
                btc_payload = {
                    "type": "candleSnapshot", 
                    "req": {
                        "coin": "BTC", 
                        "interval": self.main_timeframe, 
                        "startTime": int((datetime.now().timestamp() - 7200) * 1000),
                        "endTime": int(datetime.now().timestamp() * 1000)
                    }
                }
                btc_res = requests.post(self.info_url, json=btc_payload, timeout=5)
                if btc_res.status_code == 200:
                    candles = btc_res.json()
                    if candles and len(candles) >= 2:
                        c_start = candles[0]
                        c_end = candles[-1]
                        start_px = float(c_start['c']) if isinstance(c_start, dict) else float(c_start[4])
                        end_px = float(c_end['c']) if isinstance(c_end, dict) else float(c_end[4])
                        if start_px > 0:
                            features['btc_trend'] = (end_px - start_px) / start_px * 100

        except Exception as e:
            print(f"⚠️ 市場構造データ取得エラー: {e}")
            
        return features


    
    def get_open_interest(self):
        """現在の未決済建玉(OI)を取得"""
        try:
            payload = {"type": "metaAndAssetCtxs"}
            response = requests.post(self.info_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    state = data[0]
                    universe = state.get('universe', [])
                    asset_ctxs = state.get('assetCtxs', [])
                    
                    found_index = -1
                    for i, asset in enumerate(universe):
                        if asset['name'] == self.symbol:
                            found_index = i
                            break
                    
                    if found_index != -1 and found_index < len(asset_ctxs):
                        ctx = asset_ctxs[found_index]
                        return float(ctx.get('openInterest', 0))

            return 0.0
        except Exception as e:
            print(f"⚠️ OI取得例外エラー: {e}")
            return 0.0


if __name__ == "__main__":
    print("市場データ取得テスト")
    market = AdvancedMarketData('ETH')
    analysis = market.get_comprehensive_analysis(interval='1h') # テスト: 1hを指定
    print(f"1H RSI: {analysis['indicators'].get('rsi')}")