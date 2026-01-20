import numpy as np
from typing import Dict

class ImprovedSignalScoring:
    """
    改善版シグナルスコアリングシステム
    - 統計的に有意な閾値を使用
    - 指標間の相関（コンフルエンス）を考慮
    - ボラティリティによる動的ウェイト調整
    """
    
    def __init__(self):
        # RSI閾値
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_moderate_oversold = 40
        self.rsi_moderate_overbought = 60
        
        # ボラティリティレジーム閾値 (%)
        self.low_vol_threshold = 1.5
        self.high_vol_threshold = 5.0
        self.extreme_vol_threshold = 10.0 
    
    def calculate_normalized_macd(self, macd_hist: float, prices: np.ndarray) -> float:
        """MACDヒストグラムを価格変動率(ATR)で正規化 (-15 ~ +15)"""
        if len(prices) < 20:
            return 0.0
        
        # 簡易ATR計算 (過去20本の変動率標準偏差)
        changes = np.diff(prices[-21:])
        if len(changes) == 0: return 0.0
        
        price_std = np.std(changes)
        if price_std < 1e-9: return 0.0 # ゼロ除算対策
        
        # MACD値を標準偏差で割って正規化（Zスコアに近い考え方）
        normalized_val = macd_hist / price_std
        
        # tanhで -1 ~ 1 に収め、係数15を掛ける
        score = np.tanh(normalized_val * 0.5) * 15
        
        return float(score)
    
    def calculate_rsi_score(self, rsi: float, volatility: float) -> float:
        """ボラティリティ調整済みRSIスコア"""
        # 低ボラ時は敏感に(1.2)、高ボラ時は慎重に(0.8)反応させる
        vol_factor = 1.2 if volatility < self.low_vol_threshold else 0.8
        
        if rsi < self.rsi_oversold:        # < 30: 強い買い
            strength = (self.rsi_oversold - rsi) / self.rsi_oversold
            return min(15, strength * 15 * vol_factor)
        
        elif rsi < self.rsi_moderate_oversold: # < 40: 買い
            strength = (self.rsi_moderate_oversold - rsi) / 10.0
            return min(8, strength * 8 * vol_factor)
        
        elif rsi > self.rsi_overbought:    # > 70: 強い売り
            strength = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            return max(-15, -strength * 15 * vol_factor)
        
        elif rsi > self.rsi_moderate_overbought: # > 60: 売り
            strength = (rsi - self.rsi_moderate_overbought) / 10.0
            return max(-8, -strength * 8 * vol_factor)
        
        return 0.0
    
    def calculate_bb_score(self, bb_position: float, bb_width: float, volatility: float) -> float:
        """ボリンジャーバンドスコア (スクイーズ/エクスパンション考慮)"""
        # バンド幅による調整
        width_factor = 1.0
        if bb_width < 0.02:    width_factor = 1.5  # スクイーズ(爆発前)は重視
        elif bb_width > 0.06:  width_factor = 0.7  # 拡大しすぎは警戒
        
        # バンド位置による逆張り判定
        if bb_position < 0.2:      # 下限付近 -> 買い
            strength = (0.2 - bb_position) / 0.2
            return min(10, strength * 10 * width_factor)
            
        elif bb_position > 0.8:    # 上限付近 -> 売り
            strength = (bb_position - 0.8) / 0.2
            return max(-10, -strength * 10 * width_factor)
            
        return 0.0
    
    def calculate_trend_score(self, sma_20: float, sma_50: float, sma_200: float = None) -> float:
        """
        トレンドスコア (MAの並び順と乖離率)
        ※ volatility引数は未使用だったため削除
        """
        score = 0.0
        if sma_50 == 0: return 0.0
        
        # 短期トレンド乖離率 (%)
        diff_pct = abs(sma_20 - sma_50) / sma_50 * 100
        
        if sma_20 > sma_50: # 上昇局面
            if diff_pct > 2.0:   score += 10 # 強い上昇
            elif diff_pct > 0.5: score += 5  # 普通の上昇
        else:               # 下降局面
            if diff_pct > 2.0:   score -= 10
            elif diff_pct > 0.5: score -= 5
        
        # 長期トレンド加点 (SMA200がある場合)
        if sma_200 and sma_200 > 0:
            long_diff_pct = abs(sma_50 - sma_200) / sma_200 * 100
            if sma_50 > sma_200 and long_diff_pct > 1.0:
                score += 5
            elif sma_50 < sma_200 and long_diff_pct > 1.0:
                score -= 5
                
        return score
    
    def calculate_indicator_confluence(self, rsi_score: float, macd_score: float, bb_score: float) -> float:
        """指標間の一致度ボーナス"""
        scores = [rsi_score, macd_score, bb_score]
        
        # 全て正(買い) または 全て負(売り)
        if all(s > 0 for s in scores) or all(s < 0 for s in scores):
            avg_strength = np.mean([abs(s) for s in scores])
            return avg_strength * 0.25  # 25%ボーナス (強化)
        
        # 3つのうち2つが一致
        pos_cnt = sum(1 for s in scores if s > 0)
        neg_cnt = sum(1 for s in scores if s < 0)
        
        if pos_cnt >= 2 or neg_cnt >= 2:
            avg_strength = np.mean([abs(s) for s in scores])
            return avg_strength * 0.1   # 10%ボーナス
            
        # バラバラの場合はペナルティ
        return -5.0
    
    def calculate_dynamic_timeframe_weights(self, volatility: float, trend_strength: float) -> Dict[str, float]:
        """相場環境に応じたタイムフレームの重み付け"""
        # 1. 高ボラティリティ & 強トレンド -> 長期足(4h)に全振りしてノイズ回避
        if volatility > self.high_vol_threshold and trend_strength > 3.0:
            return {'15m': 0.10, '1h': 0.20, '4h': 0.70}
        
        # 2. 高ボラティリティのみ -> 中期足(1h-4h)重視
        elif volatility > self.high_vol_threshold:
            return {'15m': 0.15, '1h': 0.35, '4h': 0.50}
        
        # 3. 強トレンド -> 長期重視
        elif trend_strength > 3.0:
            return {'15m': 0.20, '1h': 0.30, '4h': 0.50}
        
        # 4. 低ボラ・レンジ相場 -> 短期足(15m)で回転させる
        elif volatility < self.low_vol_threshold and trend_strength < 1.0:
            return {'15m': 0.45, '1h': 0.35, '4h': 0.20}
        
        # 5. 通常時 -> バランス型
        else:
            return {'15m': 0.25, '1h': 0.35, '4h': 0.40}
    
    def calculate_comprehensive_score(self, timeframe_data: Dict[str, Dict]) -> Dict:
        """
        総合スコア計算
        DataCollector/MarketDataで計算済みの値を集約する
        """
        # 基準となる1時間足の環境認識
        base_tf = timeframe_data.get('1h', {})
        volatility = base_tf.get('volatility', 2.0)
        
        # トレンド強度 (SMA20と50の乖離)
        sma_20 = base_tf.get('sma_20', 0)
        sma_50 = base_tf.get('sma_50', 0)
        
        if sma_50 > 0:
            trend_strength = abs(sma_20 - sma_50) / sma_50 * 100
        else:
            trend_strength = 0.0
        
        # 動的重み取得
        weights = self.calculate_dynamic_timeframe_weights(volatility, trend_strength)
        
        weighted_score = 0.0
        total_weight = 0.0
        breakdown = {}
        
        for tf, data in timeframe_data.items():
            if tf not in weights: continue
            
            weight = weights[tf]
            prices = data.get('prices', np.array([]))
            
            # 各指標スコア計算
            rsi_score = self.calculate_rsi_score(data.get('rsi', 50), volatility)
            
            macd_val = data.get('macd', {})
            # 辞書型かどうかのガード
            hist = macd_val.get('histogram', 0) if isinstance(macd_val, dict) else 0
            macd_score = self.calculate_normalized_macd(hist, prices)
            
            bb_val = data.get('bollinger_bands', {})
            # AdvancedMarketDataのキー名(bollinger_bands)に合わせる
            if not bb_val: bb_val = data.get('bb', {}) # フォールバック
            
            bb_score = self.calculate_bb_score(
                bb_val.get('position', 0.5),
                bb_val.get('width', 0.04),
                volatility
            )
            
            # トレンドスコア
            t_score = self.calculate_trend_score(
                data.get('sma_20', 0),
                data.get('sma_50', 0),
                data.get('sma_200', None)
            )
            
            # コンフルエンス
            conf_bonus = self.calculate_indicator_confluence(rsi_score, macd_score, bb_score)
            
            # 合計
            tf_total = rsi_score + macd_score + bb_score + t_score + conf_bonus
            
            weighted_score += tf_total * weight
            total_weight += weight
            
            breakdown[tf] = {
                'total': tf_total,
                'weight': weight,
                'details': [rsi_score, macd_score, bb_score, t_score, conf_bonus]
            }
        
        # 正規化 (0-100)
        # weighted_scoreは概ね -50 ~ +50 の範囲になる設計
        raw_score = weighted_score / total_weight if total_weight > 0 else 0
        final_score = max(0, min(100, 50 + raw_score))
        
        # 結果整形
        if final_score > 60:   direction = 'BULLISH'
        elif final_score < 40: direction = 'BEARISH'
        else:                  direction = 'NEUTRAL'
        
        # 信頼度 (中心50からの乖離)
        confidence = min(100, abs(final_score - 50) * 2)
        
        # レジーム判定
        if volatility > self.high_vol_threshold: regime = 'VOLATILE'
        elif trend_strength > 2.0:               regime = 'TRENDING'
        else:                                    regime = 'RANGING'
        
        return {
            'signal_strength': int(final_score),
            'direction': direction,
            'confidence': int(confidence),
            'regime': regime,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'breakdown': breakdown
        }