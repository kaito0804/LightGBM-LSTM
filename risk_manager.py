# risk_manager.py 
from datetime import datetime
import json 
import os

class RiskManager:
    """
    改良版リスク管理システム
    - AI自信度に応じた柔軟なポジションサイズ
    - 段階的ポジション構築をサポート
    """
    
    # 閾値設定 (一元管理)
    CONFIDENCE_LEVELS = {
        'VERY_HIGH': 80,
        'HIGH': 60,
        'MODERATE': 40
    }
    LEVERAGE_LIMITS = {
        'VERY_HIGH': 2.8, # 目標3.0倍だがバッファを持たせて2.8で制限
        'HIGH': 1.8,      # 目標2.0倍 -> 1.8
        'MODERATE': 0.9   # 目標1.0倍 -> 0.9
    }

    def __init__(self, initial_capital=1000.0, max_daily_loss=0.10, max_leverage=1):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_daily_loss = max_daily_loss
        
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_history = []
        self.last_reset = str(datetime.now().date())
        
        self.current_position_value = 0.0 
        self.position_count = 0 

        self.state_file = "risk_state.json"
        self._load_state()
        
        print(f"🛡️ 改良版リスク管理システム")
        print(f"   レバレッジ: 最大3倍（自信度連動）")
        print(f"   初期資金: ${initial_capital:.2f}")
        print(f"   最大日次損失: {max_daily_loss*100:.0f}%")

    def _save_state(self):
        data = {
            "date": self.last_reset,
            "daily_pnl": self.daily_pnl,
            "current_capital": self.current_capital
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f)

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    saved_date = data.get("date")
                    today = str(datetime.now().date())
                    
                    if saved_date == today:
                        self.daily_pnl = data.get("daily_pnl", 0.0)
                        self.current_capital = data.get("current_capital", self.initial_capital)
                        self.last_reset = today
                        print(f"🔄 本日の損益状態を復元: ${self.daily_pnl:.2f}")
                    else:
                        print("📅 日付が変わったため損益リセット")
                        self.reset_daily_stats()
            except Exception as e:
                print(f"⚠️ 状態読み込みエラー: {e}")

    def calculate_position_size_by_confidence(
        self, 
        capital: float,
        entry_price: float,
        confidence: float,
        existing_position_value: float = 0.0,
        stop_loss_percent: float = 3.0,
        max_available_cash: float = None) -> dict:
        
        print(f"\n🔍 [ポジションサイズ計算]")
        print(f"資金(Equity): ${capital:.2f}")
        print(f"既存ポジション: ${existing_position_value:.2f}")
        print(f"AI自信度: {confidence:.0f}/100")
        
        # === ステップ1: 自信度に応じたレバレッジ倍率決定 ===
        if confidence >= self.CONFIDENCE_LEVELS['VERY_HIGH']:
            target_leverage = 3.0  
            risk_level = "VERY_HIGH_CONFIDENCE"
            reasoning = "超高自信度(80+) - 最大レバレッジ3倍適用"
        elif confidence >= self.CONFIDENCE_LEVELS['HIGH']:
            target_leverage = 2.0 
            risk_level = "HIGH_CONFIDENCE"
            reasoning = "高自信度(60+) - レバレッジ2倍適用"
        elif confidence >= self.CONFIDENCE_LEVELS['MODERATE']:
            target_leverage = 1.0 
            risk_level = "MODERATE_CONFIDENCE"
            reasoning = "中自信度(40+) - レバレッジ1倍維持"
        else:
            target_leverage = 0.5 
            risk_level = "LOW_CONFIDENCE"
            reasoning = "低自信度 - ポジション縮小"

        # === ステップ2: 目標ポジション総額の計算 ===
        target_position_value = capital * target_leverage
        max_new_position_value = target_position_value - existing_position_value
        
        print(f"\n📊 戦略設定:")
        print(f"   目標レバレッジ: {target_leverage}倍")
        print(f"   目標総ポジション: ${target_position_value:.2f}")
        print(f"   追加可能枠: ${max_new_position_value:.2f}")
        
        if max_new_position_value <= 0:
            return {
                'size': 0.0, 'position_value': 0.0,
                'risk_level': risk_level,
                'reasoning': f"既存ポジションが目標レバレッジ({target_leverage}x)に到達済み"
            }
        
        # === ステップ3: 1回の注文サイズを制限 (分割エントリー) ===
        order_value_limit = capital * 1.0
        new_position_value = min(max_new_position_value, order_value_limit)
        
        # === ステップ4: 数量計算 ===
        position_size = new_position_value / entry_price if entry_price > 0 else 0
        
        # 最小サイズチェック (Hyperliquidは約$10〜12が必要)
        min_order_usd = 12.0
        min_size = max(min_order_usd / entry_price, 0.004)
        
        if position_size < min_size:
            # 自信度が高ければ最小サイズまで引き上げる
            if confidence >= 50:
                print(f"   ⚠️ サイズ不足だが最小サイズへ切り上げ")
                position_size = min_size
                new_position_value = position_size * entry_price
            else:
                return {
                    'size': 0.0, 'position_value': 0.0,
                    'risk_level': risk_level,
                    'reasoning': "サイズ不足かつ自信度不足"
                }

        # === ステップ5: 損失許容額チェック ===
        sl_distance = entry_price * (stop_loss_percent / 100)
        potential_loss = position_size * sl_distance
        
        # 残りの日次損失許容枠を計算
        current_loss_ratio = abs(self.daily_pnl / self.initial_capital) if self.daily_pnl < 0 else 0
        remaining_risk_pct = max(0, self.max_daily_loss - current_loss_ratio)
        max_allowed_loss = capital * remaining_risk_pct
        
        if potential_loss > max_allowed_loss:
            print(f"   🛑 リスク許容額超過: 損失予定${potential_loss:.2f} > 許容${max_allowed_loss:.2f}")
            if sl_distance > 0:
                adjusted_size = max_allowed_loss / sl_distance
                position_size = adjusted_size
                new_position_value = position_size * entry_price
                reasoning += " (リスク許容額に合わせて縮小)"
            else:
                position_size = 0
            
        print(f"\n✅ 最終決定:")
        print(f"   注文サイズ: {position_size:.4f} ETH (${new_position_value:.2f})")
        print(f"   レバレッジ効果: 資金の{new_position_value/capital:.2f}倍を追加")

        return {
            'size': round(position_size, 4),
            'position_value': round(new_position_value, 2),
            'risk_level': risk_level,
            'reasoning': reasoning
        }
    
    def calculate_position_size(self, capital, risk_percent, entry_price, stop_loss_percent=2.0):
        """互換性用"""
        return self.calculate_position_size_by_confidence(
            capital, entry_price, 60, 0, stop_loss_percent
        )['size']
    
    def update_position_tracking(self, position_value: float, action: str = "ADD"):
        if action == "ADD":
            self.current_position_value += position_value
            self.position_count += 1
            print(f"📊 ポジション追加: ${position_value:.2f}")
            print(f"   現在の総ポジション: ${self.current_position_value:.2f}")
        elif action == "CLOSE":
            self.current_position_value = 0
            self.position_count = 0
            print(f"📊 ポジションクローズ")
    
    def should_add_position(self, confidence: float, current_position_value: float) -> bool:
        """追加ポジションを取るべきか判定"""
        if current_position_value == 0: return True
        position_ratio = current_position_value / self.current_capital
        
        # 最大レバレッジ(3倍)に近い場合は絶対停止
        if position_ratio >= self.LEVERAGE_LIMITS['VERY_HIGH']: 
            return False
        
        # 自信度ごとの許容レバレッジチェック
        if confidence >= self.CONFIDENCE_LEVELS['VERY_HIGH']: 
            return position_ratio < self.LEVERAGE_LIMITS['VERY_HIGH']
        elif confidence >= self.CONFIDENCE_LEVELS['HIGH']: 
            return position_ratio < self.LEVERAGE_LIMITS['HIGH']
        elif confidence >= self.CONFIDENCE_LEVELS['MODERATE']: 
            return position_ratio < self.LEVERAGE_LIMITS['MODERATE']
        else: 
            return False
    
    def calculate_stop_loss(self, entry_price, side, atr=None, percent=3.0):
        safe_percent = min(percent, 5.0) # 最大5%まで
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
        if today != self.last_reset:
            self.reset_daily_stats()
        
        if self.daily_pnl < 0:
            daily_loss_ratio = abs(self.daily_pnl / self.initial_capital)
            if daily_loss_ratio >= self.max_daily_loss:
                print(f"🛑 日次損失限度到達: {daily_loss_ratio*100:.1f}%")
                return False
        return True
    
    def update_daily_pnl(self, pnl):
        self.daily_pnl += pnl
        self.total_pnl += pnl
        self.current_capital += pnl
        self._save_state()
    
    def reset_daily_stats(self):
        self.daily_pnl = 0.0
        self.trade_history = []
        self.last_reset = str(datetime.now().date())
        self._save_state()
    
    def get_risk_summary(self, entry_price, position_size, stop_loss, take_profit, leverage):
        risk_amount = abs(entry_price - stop_loss) * position_size
        reward_amount = abs(entry_price - take_profit) * position_size
        risk_pct = (risk_amount / self.initial_capital) * 100
        reward_pct = (reward_amount / self.initial_capital) * 100
        
        return {
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'risk_percentage': risk_pct,
            'reward_percentage': reward_pct,
            'risk_reward_ratio': reward_amount / risk_amount if risk_amount > 0 else 0
        }

    def print_risk_status(self):
        print("\n" + "="*60)
        print("🛡️ リスク管理状況")
        print(f"資金: ${self.current_capital:.2f}")
        print(f"本日損益: ${self.daily_pnl:+.2f}")
        print(f"ポジション: ${self.current_position_value:.2f}")
        print("="*60 + "\n")