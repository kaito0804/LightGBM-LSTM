# advanced_market_data.py (修正版)
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
            # Hyperliquid API v2 は通常 dict のリスト [{'t':..., 'o':...}, ...] を返す
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
                # 必要に応じてリスト形式のパースもここに追加可能
            
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
        # データ不足チェック
        # RSIは過去の影響を引きずるため、期間の3〜5倍のデータがないと精度が出ない
        if len(prices) < period + 2:
            return 50.0
        
        prices_series = pd.Series(prices)
        deltas = prices_series.diff()
        
        # 上昇幅と下落幅の分離
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        # Wilder's Smoothing (alpha = 1/period は com = period-1 と等価)
        # adjust=False にすることで、再帰的な計算（Wilder式）を再現
        avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # 最新値の取得
        curr_gain = avg_gain.iloc[-1]
        curr_loss = avg_loss.iloc[-1]
        
        # ゼロ除算対策 (下落幅がほぼ0なら最強のRSI=100)
        if curr_loss < 1e-10:
            return 100.0 if curr_gain > 0 else 50.0
        
        rs = curr_gain / curr_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)



    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        MACD計算（Pandas標準）
        """
        # シグナルラインまで計算するには (slow + signal) 以上のデータが最低限必要
        # EMAの収束安定性のために +10 程度の余裕を持たせるのは適切
        min_required = slow + signal + 10

        if len(prices) < min_required:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        prices_series = pd.Series(prices)
        
        # adjust=False は標準的なEMA（再帰的計算）
        ema_fast = prices_series.ewm(span=fast, adjust=False).mean()
        ema_slow = prices_series.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        # 安全な値取得ヘルパー (NaNの場合は0.0を返す)
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
        # データ不足チェック
        if len(prices) < period:
            # データが足りない場合は現在価格に収束させる（安全策）
            current = float(prices[-1]) if len(prices) > 0 else 0.0
            return {
                'upper': current,
                'middle': current,
                'lower': current,
                'position': 0.5, # 中立
                'width': 0.0     # バンド幅なし
            }
        
        prices_series = pd.Series(prices)
        
        # 移動平均と標準偏差
        # ddof=0 は母集団標準偏差。多くのチャートツールと一致させるため維持
        sma = prices_series.rolling(window=period).mean()
        std = prices_series.rolling(window=period).std(ddof=0)
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        # 最新値の取得（NaNチェック付きヘルパー）
        def get_val(series):
            val = series.iloc[-1]
            return 0.0 if pd.isna(val) else float(val)

        upper_val = get_val(upper)
        lower_val = get_val(lower)
        sma_val = get_val(sma)
        current_price = float(prices[-1])
        
        # %B (Position) の計算: バンド内のどこにいるか (0=下限, 0.5=中央, 1=上限)
        if upper_val > lower_val:
            position = (current_price - lower_val) / (upper_val - lower_val)
        else:
            position = 0.5
            
        # Bandwidth の計算: バンド幅の広さ（ボラティリティの指標）
        if sma_val != 0:
            width = (upper_val - lower_val) / sma_val
        else:
            width = 0.0
        
        return {
            'upper': upper_val,
            'middle': sma_val,
            'lower': lower_val,
            'position': float(position),
            'width': float(width)
        }



    def calculate_atr(self, df, period=14):
        """
        ATR計算 - Wilder's平滑化方式
        """
        # データ不足チェック
        if len(df) < period:
            if len(df) > 0:
                return float((df['high'] - df['low']).mean())
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # TR (True Range) の計算
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        
        # 3つの中で最大のものを採用
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Wilder's Smoothing
        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        
        # 最新値の取得 (NaN対策)
        val = atr.iloc[-1]
        return 0.0 if pd.isna(val) else float(val)



    def get_comprehensive_analysis(self):
        """
        総合的な市場分析（改善版スコアリング統合）
        ✅ 重複計算の排除とロジック整理
        """
        analysis = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframes': {},
            'indicators': {}, # 1h足の指標をここに格納
            'trend': {},
            'signal_strength': 0,
            'recommendation': 'HOLD',
            'volatility': 0.0,
            'sentiment': 'NEUTRAL',
            'market_structure': {'orderbook_imbalance': 0.0, 'btc_trend': 0.0}
        }
        
        scoring_data = {}
        
        # タイムフレームごとの取得設定
        # SMA200や長期トレンド判定に必要な長さを確保
        timeframe_config = {
            '15m': {'limit': 300}, 
            '1h': {'limit': 400},  
            '4h': {'limit': 500}   
        }

        # --- 1. データ収集とテクニカル計算 ---
        for tf, config in timeframe_config.items():
            df = self.get_ohlcv(timeframe=tf, limit=config['limit'])
            
            # データ不足時はスキップ
            if df is None or len(df) < 50:
                print(f"⚠️ {tf} データ不足のため分析スキップ")
                continue
            
            prices = df['close'].values
            
            # テクニカル指標計算 (ここで1回だけ計算)
            rsi = self.calculate_rsi(prices)
            macd = self.calculate_macd(prices)
            bb = self.calculate_bollinger_bands(prices)
            atr = self.calculate_atr(df)
            
            # SMA計算
            sma_20 = float(np.mean(prices[-20:]))
            sma_50 = float(np.mean(prices[-50:]))
            sma_200 = float(np.mean(prices[-200:])) if len(prices) >= 200 else None
            
            # トレンド判定
            trend_dir = "上昇" if sma_20 > sma_50 else "下降"
            trend_str = abs(sma_20 - sma_50) / sma_50 * 100 if sma_50 != 0 else 0
            
            # ボラティリティ
            vol_period = min(20, len(prices))
            volatility = float(np.std(prices[-vol_period:]) / np.mean(prices[-vol_period:]) * 100)

            # 極端なボラティリティの場合は警告 (スキップはしないがログに残す)
            if volatility > self.scorer.extreme_vol_threshold:
                print(f"⚠️ {tf} ボラティリティ過大: {volatility:.2f}%")
            
            # スコアリング用データ蓄積
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
            
            # 分析結果格納
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
                # 後続処理のために生の価格配列も保持
                'prices': prices,
                'df_summary': df.iloc[-1].to_dict()
            }
            analysis['timeframes'][tf] = tf_data
            
            # --- 設定したメイン時間軸（15m）のデータを優先採用する ---
            if tf == self.main_timeframe:
                analysis['indicators'] = {
                    'rsi': rsi,
                    'macd': macd,
                    'bollinger': bb,
                    'atr': atr
                }
                analysis['trend'] = {'direction': trend_dir, 'strength': trend_str}

        # --- 2. 総合スコアリング ---
        if scoring_data:
            scoring_result = self.scorer.calculate_comprehensive_score(scoring_data)
            
            analysis['signal_strength'] = scoring_result['signal_strength']
            analysis['volatility'] = scoring_result['volatility']
            analysis['sentiment'] = scoring_result['direction']
            
            # 推奨アクションの決定ロジック
            strength = scoring_result['signal_strength']
            direction = scoring_result['direction']
            
            if strength > 70:
                rec = 'STRONG_BUY' if direction == 'BULLISH' else 'STRONG_SELL'
            elif strength > 55:
                rec = 'BUY' if direction == 'BULLISH' else 'SELL'
            elif strength < 30: # 弱気シグナルが強い場合
                rec = 'STRONG_SELL' if direction == 'BEARISH' else 'STRONG_BUY' # (※逆張りの可能性もあるが、通常はトレンドフォロー)
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
        現在価格を取得
        ✅ 修正: ローソク足ではなく 'allMids' (板の中値) を使用して高速化・リアルタイム化
        """
        try:
            # Hyperliquidの軽量エンドポイント 'allMids' を使用
            payload = {"type": "allMids"}
            response = requests.post(self.info_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # self.symbol (例: ETH) の価格を取り出す
                if self.symbol in data:
                    return float(data[self.symbol])
            
            # 取得失敗時
            raise ValueError(f"allMids取得失敗: Status {response.status_code}")

        except Exception as e:
            print(f"⚠️ 現在価格取得エラー: {e} -> データ取得失敗")
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
            # 1. 板情報の不均衡 (Orderbook Imbalance)
            payload = {"type": "l2Snapshot", "coin": self.symbol}
            response = requests.post(self.info_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # levelsキーが存在し、かつデータがあるか確認
                if 'levels' in data and len(data['levels']) >= 2:
                    bids = data['levels'][0]
                    asks = data['levels'][1]
                    
                    if bids and asks:
                        # 上位10本の板厚
                        bid_vol = sum([float(x['sz']) for x in bids[:10]])
                        ask_vol = sum([float(x['sz']) for x in asks[:10]])
                        
                        total_vol = bid_vol + ask_vol
                        if total_vol > 0:
                            # 1に近いほど買い圧、-1に近いほど売り圧
                            features['orderbook_imbalance'] = (bid_vol - ask_vol) / total_vol
            
            # 2. BTCトレンド (BTC相関)
            # 自分がBTCでない場合のみ取得
            if self.symbol != 'BTC':
                btc_payload = {
                    "type": "candleSnapshot", 
                    "req": {
                        "coin": "BTC", 
                        "interval": self.main_timeframe, 
                        "startTime": int((datetime.now().timestamp() - 7200) * 1000), # 2時間前
                        "endTime": int(datetime.now().timestamp() * 1000)
                    }
                }
                btc_res = requests.post(self.info_url, json=btc_payload, timeout=5)
                if btc_res.status_code == 200:
                    candles = btc_res.json()
                    if candles and len(candles) >= 2:
                        # APIの返却形式(dict or list)に対応
                        c_start = candles[0]
                        c_end = candles[-1]
                        
                        start_px = float(c_start['c']) if isinstance(c_start, dict) else float(c_start[4])
                        end_px = float(c_end['c']) if isinstance(c_end, dict) else float(c_end[4])
                        
                        if start_px > 0:
                            features['btc_trend'] = (end_px - start_px) / start_px * 100

        except Exception as e:
            # 特徴量取得失敗は致命的エラーにしない（0埋めで続行）
            print(f"⚠️ 市場構造データ取得エラー: {e}")
            
        return features


    
    def get_open_interest(self):
        """
        【診断モード】現在の未決済建玉(OI)を取得
        """
        try:
            payload = {"type": "metaAndAssetCtxs"}
            response = requests.post(self.info_url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    state = data[0]
                    universe = state.get('universe', [])
                    asset_ctxs = state.get('assetCtxs', [])
                    
                    # 診断ログ: データ件数を表示
                    # print(f"DEBUG: Universe len={len(universe)}, AssetCtxs len={len(asset_ctxs)}")
                    
                    # シンボル検索
                    found_index = -1
                    for i, asset in enumerate(universe):
                        if asset['name'] == self.symbol:
                            found_index = i
                            break
                    
                    if found_index != -1:
                        # シンボルは見つかった
                        if found_index < len(asset_ctxs):
                            ctx = asset_ctxs[found_index]
                            oi = float(ctx.get('openInterest', 0))
                            # print(f"DEBUG: Symbol {self.symbol} found at {found_index}. OI={oi}")
                            return oi
                        else:
                            # ここが原因かチェック
                            print(f"⚠️ OI診断エラー: インデックス超過 (Symbol: {self.symbol}, Index: {found_index}, CtxLen: {len(asset_ctxs)})")
                            return 0.0
                    else:
                        # シンボルが見つからない
                        print(f"⚠️ OI診断エラー: シンボルが見つかりません (Target: {self.symbol})")
                        # 念のため似た名前がないか探す
                        # similar = [a['name'] for a in universe if 'ETH' in a['name']]
                        # print(f"   (参考) 'ETH'を含む銘柄: {similar}")
                        return 0.0

            return 0.0
        except Exception as e:
            print(f"⚠️ OI取得例外エラー: {e}")
            return 0.0


if __name__ == "__main__":
    print("="*70)
    print("📊 市場データ取得テスト (修正版)")
    print("="*70)
    
    market = AdvancedMarketData('ETH')
    
    try:
        price = market.get_current_price()
        print(f"\n現在価格: ${price:.2f}")
        
        print("\n--- 1時間足データ取得テスト ---")
        df = market.get_ohlcv(timeframe='1h', limit=10)
        if df is not None:
            print(df.tail())
        
        print("\n--- 総合市場分析テスト ---")
        analysis = market.get_comprehensive_analysis()
        
        print(f"\nシンボル: {analysis['symbol']}")
        print(f"総合シグナル強度: {analysis['signal_strength']}/100")
        print(f"推奨: {analysis['recommendation']}")
        print(f"センチメント: {analysis['sentiment']}")
        
        if 'indicators' in analysis:
            print(f"\n主要指標:")
            print(f"  RSI: {analysis['indicators'].get('rsi', 0):.2f}")
            print(f"  MACD: {analysis['indicators'].get('macd', {}).get('histogram', 0):.4f}")
            print(f"  BB位置: {analysis['indicators'].get('bollinger', {}).get('position', 0):.2f}")
            
    except Exception as e:
        print(f"エラー発生: {e}")
        import traceback
        traceback.print_exc()