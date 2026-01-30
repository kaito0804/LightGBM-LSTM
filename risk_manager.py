# risk_manager.py (Full Update for .env support)
from datetime import datetime
import json 
import os
from dotenv import load_dotenv

load_dotenv()

class RiskManager:
    """
    改良版リスク管理システム (.env対応版)
    """
    
    def __init__(self, initial_capital=1000.0, max_leverage=1):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # .envから設定を読み込み
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 0.10))
        
        # 自信度閾値
        self.confidence_levels = {
            'VERY_HIGH': int(os.getenv('CONFIDENCE_VERY_HIGH', 80)),
            'HIGH': int(os.getenv('CONFIDENCE_HIGH', 60)),
            'MODERATE': int(os.getenv('CONFIDENCE_MODERATE', 40))
        }
        
        # レバレッジ設定
        self.leverage_limits = {
            'VERY_HIGH': float(os.getenv('LEVERAGE_VERY_HIGH', 2.8)),
            'HIGH': float(os.getenv('LEVERAGE_HIGH', 1.8)),
            'MODERATE': float(os.getenv('LEVERAGE_MODERATE', 0.9)),
            'LOW': float(os.getenv('LEVERAGE_LOW', 0.5))
        }

        # 日次管理用
        self.start_of_day_capital = initial_capital
        self.daily_pnl = 0.0
        self.last_reset_date = str(datetime.now().date())
        
        self.current_position_value = 0.0 
        self.state_file = "risk_state.json"
        self._load_state()
        
        print(f"🛡️ リスク管理システム初期化")
        print(f"   日次許容損失: {self.max_daily_loss*100}%")
        print(f"   自信度閾値: {self.confidence_levels}")

    def _save_state(self):
        """状態保存"""
        data = {
            "date": self.last_reset_date,
            "start_of_day_capital": self.start_of_day_capital,
            "current_capital": self.current_capital
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"⚠️ リスク状態保存エラー: {e}")

    def _load_state(self):
        """状態復元"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    saved_date = data.get("date")
                    today = str(datetime.now().date())
                    
                    if saved_date == today:
                        self.start_of_day_capital = data.get("start_of_day_capital", self.initial_capital)
                        self.current_capital = data.get("current_capital", self.initial_capital)
                        self.last_reset_date = today
                        self._recalc_daily_pnl()
                    else:
                        print("📅 日付変更検知: 損益リセット")
                        self.reset_daily_stats(new_capital=data.get("current_capital", self.initial_capital))
            except Exception as e:
                print(f"⚠️ 状態読み込みエラー: {e}")

    def _recalc_daily_pnl(self):
        self.daily_pnl = self.current_capital - self.start_of_day_capital

    def reset_daily_stats(self, new_capital=None):
        if new_capital is not None:
            self.current_capital = new_capital
        self.start_of_day_capital = self.current_capital
        self.daily_pnl = 0.0
        self.last_reset_date = str(datetime.now().date())
        self._save_state()

    def sync_account_state(self, current_equity: float, position_value: float):
        today = str(datetime.now().date())
        if today != self.last_reset_date:
            print(f"📅 日付変更リセット実行 ({self.last_reset_date} -> {today})")
            self.reset_daily_stats(new_capital=current_equity)
            return

        if abs(self.current_capital - current_equity) > 0.1:
            self.current_capital = current_equity
            self._recalc_daily_pnl()
            self._save_state()
            
        self.current_position_value = position_value

    def calculate_position_size_by_confidence(
        self, 
        capital: float,
        entry_price: float,
        confidence: float,
        existing_position_value: float = 0.0,
        stop_loss_percent: float = 3.0,
        max_available_cash: float = None) -> dict:
        
        # === ステップ1: 自信度に応じたレバレッジ倍率決定 (変数化) ===
        if confidence >= self.confidence_levels['VERY_HIGH']:
            target_leverage = self.leverage_limits['VERY_HIGH']
            risk_level = "VERY_HIGH_CONFIDENCE"
            reasoning = f"超高自信度({self.confidence_levels['VERY_HIGH']}+) - MaxLev {target_leverage}x"
        elif confidence >= self.confidence_levels['HIGH']:
            target_leverage = self.leverage_limits['HIGH']
            risk_level = "HIGH_CONFIDENCE"
            reasoning = f"高自信度({self.confidence_levels['HIGH']}+) - Lev {target_leverage}x"
        elif confidence >= self.confidence_levels['MODERATE']:
            target_leverage = self.leverage_limits['MODERATE']
            risk_level = "MODERATE_CONFIDENCE"
            reasoning = f"中自信度({self.confidence_levels['MODERATE']}+) - Lev {target_leverage}x"
        else:
            target_leverage = self.leverage_limits['LOW']
            risk_level = "LOW_CONFIDENCE"
            reasoning = "低自信度 - ポジション縮小"

        # === ステップ2: 目標ポジション総額の計算 ===
        target_position_value = capital * target_leverage
        max_new_position_value = target_position_value - existing_position_value
        
        if max_new_position_value <= 0:
            return {
                'size': 0.0, 'position_value': 0.0,
                'risk_level': risk_level,
                'reasoning': f"既存Posが目標({target_leverage}x)到達済"
            }
        
        order_value_limit = capital * 1.0 # 安全のため1回の最大注文は元本等倍まで
        new_position_value = min(max_new_position_value, order_value_limit)
        
        position_size = new_position_value / entry_price if entry_price > 0 else 0
        
        # 最小サイズチェック
        min_order_usd = 12.0
        min_size = max(min_order_usd / entry_price, 0.004)
        
        if position_size < min_size:
            if confidence >= 50:
                position_size = min_size
                new_position_value = position_size * entry_price
            else:
                return {
                    'size': 0.0, 'position_value': 0.0,
                    'risk_level': risk_level,
                    'reasoning': "サイズ不足かつ自信度不足"
                }

        # === ステップ3: 損失許容額チェック ===
        sl_distance = entry_price * (stop_loss_percent / 100)
        potential_loss = position_size * sl_distance
        
        current_loss_amount = abs(self.daily_pnl) if self.daily_pnl < 0 else 0
        max_loss_amount = self.initial_capital * self.max_daily_loss
        remaining_loss_allowance = max(0, max_loss_amount - current_loss_amount)
        
        if potential_loss > remaining_loss_allowance:
            print(f"   🛑 リスク許容額超過: 予定損失${potential_loss:.2f} > 残り許容${remaining_loss_allowance:.2f}")
            if sl_distance > 0:
                adjusted_size = remaining_loss_allowance / sl_distance
                position_size = adjusted_size
                new_position_value = position_size * entry_price
                reasoning += " (日次損失許容調整)"
            else:
                position_size = 0

        return {
            'size': round(position_size, 4),
            'position_value': round(new_position_value, 2),
            'risk_level': risk_level,
            'reasoning': reasoning
        }

    def update_position_tracking(self, position_value: float, action: str = "ADD"):
        # 互換性維持
        if action == "ADD":
            self.current_position_value += position_value
        elif action == "CLOSE":
            self.current_position_value = 0

    def should_add_position(self, confidence: float, current_position_value: float) -> bool:
        """追加ポジションを取るべきか判定"""
        if current_position_value == 0: return True
        cap = self.current_capital if self.current_capital > 0 else 1.0
        position_ratio = current_position_value / cap
        
        # 変数化された閾値を使用
        limit_very_high = self.leverage_limits['VERY_HIGH']
        limit_high = self.leverage_limits['HIGH']
        limit_moderate = self.leverage_limits['MODERATE']

        if position_ratio >= limit_very_high: return False
        
        if confidence >= self.confidence_levels['VERY_HIGH']: 
            return position_ratio < limit_very_high
        elif confidence >= self.confidence_levels['HIGH']: 
            return position_ratio < limit_high
        elif confidence >= self.confidence_levels['MODERATE']: 
            return position_ratio < limit_moderate
        else: 
            return False
            
    def calculate_stop_loss(self, entry_price, side, atr=None, percent=3.0):
        safe_percent = min(percent, 10.0) 
        if side.upper() == "LONG":
            stop_loss = entry_price * (1 - safe_percent / 100)
        else:
            stop_loss = entry_price * (1 + safe_percent / 100)
        return round(stop_loss, 2)

    def calculate_take_profit(self, entry_price, stop_loss_price, risk_reward_ratio=1.5):
        risk = abs(entry_price - stop_loss_price)
        reward = risk * risk_reward_ratio
        if entry_price > stop_loss_price:
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        return round(take_profit, 2)

    def check_daily_loss_limit(self):
        today = str(datetime.now().date())
        if today != self.last_reset_date:
            self._recalc_daily_pnl()

        if self.daily_pnl < 0:
            daily_loss_ratio = abs(self.daily_pnl / self.initial_capital)
            if daily_loss_ratio >= self.max_daily_loss:
                print(f"🛑 日次損失限度到達: {daily_loss_ratio*100:.1f}% (PnL: ${self.daily_pnl:.2f})")
                return False
        return True