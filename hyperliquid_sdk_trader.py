# hyperliquid_sdk_trader.py
# Hyperliquid公式SDK使用版(完全版)

import os
import sys
import json
import math
import requests
import traceback
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange

load_dotenv(override=True)

class HyperliquidSDKTrader:
    """
    Hyperliquid公式SDK使用版トレーダー
    - 成行・指値注文、ポジション管理、口座情報取得をサポート
    """
    
    def __init__(self, vault_address=None):
        print("🔍 秘密鍵を検索中...")
        
        # 秘密鍵取得(優先順位を明確化)
        keys = {
            "METAMASK_PRIVATE_KEY": os.getenv("METAMASK_PRIVATE_KEY"),
            "TESTNET_SECRET_KEY": os.getenv("TESTNET_SECRET_KEY"),
            "HYPERLIQUID_TEST_PRIVATE_KEY": os.getenv("HYPERLIQUID_TEST_PRIVATE_KEY"),
            "HYPERLIQUID_PRIVATE_KEY": os.getenv("HYPERLIQUID_PRIVATE_KEY")
        }
        
        # 設定状況の表示
        for k, v in keys.items():
            print(f"   {k}: {'設定あり' if v else '未設定'}")
            
        # 優先順位に従ってキーを選択
        private_key = keys["METAMASK_PRIVATE_KEY"] or keys["TESTNET_SECRET_KEY"] or \
                      keys["HYPERLIQUID_TEST_PRIVATE_KEY"] or keys["HYPERLIQUID_PRIVATE_KEY"]
        
        if not private_key:
            raise ValueError("❌ 秘密鍵が.envに設定されていません。")
        
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key
        
        # アカウント作成
        self.account = Account.from_key(private_key)
        self.address = self.account.address
        
        # ネットワーク設定
        network = os.getenv("NETWORK", "testnet").lower()
        if network == "mainnet":
            self.api_base = "https://api.hyperliquid.xyz"
            print("🚀 MAINNETモードで起動中...")
        else:
            self.api_base = "https://api.hyperliquid-testnet.xyz"
            print("🛡️ TESTNETモードで起動中...")
        
        # Vaultアドレス
        self.vault_address = vault_address or os.getenv("HYPERLIQUID_VAULT_ADDRESS")
        
        # API初期化
        self.info = Info(base_url=self.api_base, skip_ws=True)
        self.exchange = Exchange(
            self.account,
            base_url=self.api_base,
            vault_address=self.vault_address
        )
        
        print(f"\n✅ Hyperliquid Python SDK接続完了")
        print(f"   署名アドレス: {self.address}")
        if self.vault_address:
            print(f"   Vaultアドレス: {self.vault_address}")
    
    # =========================================================================
    # 情報取得メソッド
    # =========================================================================
    
    def get_user_state(self):
        """ユーザー状態取得 (Perps)"""
        try:
            target = self.vault_address or self.address
            return self.info.user_state(target)
        except Exception as e:
            print(f"❌ ユーザー状態取得エラー: {e}")
            return None
    
    def get_spot_balance(self):
        """Spot残高取得"""
        try:
            target = self.vault_address or self.address
            return self.info.spot_user_state(target)
        except Exception:
            # SDKメソッドがない場合のフォールバック
            try:
                url = f"{self.api_base}/info"
                payload = {"type": "spotClearinghouseState", "user": target}
                res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
                return res.json() if res.status_code == 200 else None
            except:
                return None

    def get_current_price(self, symbol="ETH"):
        """現在価格取得 (allMids)"""
        try:
            all_mids = self.info.all_mids()
            if symbol in all_mids:
                return float(all_mids[symbol])
            print(f"⚠️ {symbol}の価格が見つかりません")
            return None
        except Exception as e:
            print(f"❌ 価格取得エラー: {e}")
            return None

    def get_orderbook_snapshot(self, symbol):
        """板情報のスナップショットを取得"""
        try:
            return self.info.l2_snapshot(symbol)
        except Exception as e:
            print(f"❌ 板情報取得エラー: {e}")
            return None

    # =========================================================================
    # ユーティリティ
    # =========================================================================

    def _round_size(self, size, symbol):
        """数量の丸め処理 (簡易版)"""
        decimals = 4 if "ETH" in symbol else 3
        factor = 10 ** decimals
        return math.floor(size * factor) / factor

    def _round_price(self, price):
        """価格の丸め処理 (有効数字5桁)"""
        if price == 0: return 0.0
        return float(f"{price:.5g}")

    # =========================================================================
    # 注文・ポジション操作
    # =========================================================================

    def cancel_all_orders(self, symbol):
        """指定シンボルのオープン注文をすべてキャンセル"""
        try:
            open_orders = self.info.open_orders(self.address)
            cancelled = 0
            for order in open_orders:
                if order['coin'] == symbol:
                    self.exchange.cancel(symbol, order['oid'])
                    cancelled += 1
            
            if cancelled > 0:
                print(f"🗑️ 既存注文 {cancelled}件をキャンセルしました")
        except Exception as e:
            print(f"⚠️ 注文キャンセル失敗 (影響なし): {e}")

    def place_order(self, symbol, is_buy, size, price=None, order_type="market", reduce_only=False):
        """
        基本注文メソッド (主に成行用 / CLI用)
        """
        try:
            print(f"📤 注文送信中 ({order_type})...")
            
            if order_type == "market":
                result = self.exchange.market_open(
                    symbol, is_buy, size, None, 0.05
                )
            else:
                # 指値の場合は place_limit_order の使用を推奨するが、互換性のため残す
                if price is None:
                    price = self.get_current_price(symbol)
                
                result = self.exchange.order(
                    symbol, is_buy, size, price,
                    {"limit": {"tif": "Gtc"}},
                    reduce_only=reduce_only
                )
            
            if result and result.get('status') == 'ok':
                print(f"✅ 注文成功: {symbol} {'BUY' if is_buy else 'SELL'} {size}")
                return result
            else:
                print(f"❌ 注文失敗: {result}")
                return None
                
        except Exception as e:
            print(f"❌ 注文エラー: {e}")
            traceback.print_exc()
            return None

    def place_limit_order(self, symbol, is_buy, size, time_in_force="Gtc", aggressive=True):
        """
        【ボット用】高度な指値注文
        - 板情報を確認して価格を決定
        - AggressiveモードならIOC(即時約定orキャンセル)を強制
        - 既存注文の自動キャンセル機能付き
        """
        try:
            # 1. 既存注文をクリア
            self.cancel_all_orders(symbol)

            # 2. 板情報取得
            snapshot = self.get_orderbook_snapshot(symbol)
            if not snapshot or 'levels' not in snapshot:
                print("⚠️ 板情報取得失敗のため注文中止")
                return None

            bids = snapshot['levels'][0]
            asks = snapshot['levels'][1]
            if not bids or not asks:
                return None

            best_bid = float(bids[0]['px'])
            best_ask = float(asks[0]['px'])
            
            # 3. 価格決定 (Aggressiveなら相手の板にぶつける)
            if is_buy:
                raw_price = best_ask if aggressive else best_bid
            else:
                raw_price = best_bid if aggressive else best_ask
            
            price = self._round_price(raw_price)
            size = self._round_size(size, symbol)

            if size <= 0:
                print(f"⚠️ 数量不足: {size}")
                return None

            # 4. TIF設定 (AggressiveならIOC強制)
            final_tif = "Ioc" if aggressive else time_in_force

            print(f"🛡️ 指値注文 ({final_tif}): {symbol} {'BUY' if is_buy else 'SELL'} {size} @ {price}")

            # 5. 実行
            return self.exchange.order(
                symbol, is_buy, size, price,
                {"limit": {"tif": final_tif}}
            )
        except Exception as e:
            print(f"❌ 指値注文エラー: {e}")
            return None

    def close_position(self, symbol):
        """ポジションを全決済"""
        try:
            state = self.get_user_state()
            if not state: return None
            
            positions = state.get('assetPositions', [])
            for pos in positions:
                p_data = pos.get('position', {})
                if p_data.get('coin') == symbol:
                    szi = float(p_data.get('szi', 0))
                    if szi == 0:
                        print(f"ℹ️ {symbol} のポジションはありません")
                        return None
                    
                    is_long = szi > 0
                    size = abs(szi)
                    
                    print(f"📉 {symbol} クローズ: {'LONG' if is_long else 'SHORT'} {size}")
                    
                    # SDKの便利メソッドを使用
                    result = self.exchange.market_close(symbol, size)
                    
                    if result and result.get('status') == 'ok':
                        print("✅ クローズ成功")
                        return result
                    else:
                        print(f"❌ クローズ失敗: {result}")
                        return None
            
            print(f"ℹ️ {symbol} のポジションが見つかりません")
            return None
            
        except Exception as e:
            print(f"❌ クローズエラー: {e}")
            return None

    def print_account_status(self):
        """アカウント状況の表示"""
        print("\n" + "="*70)
        print("📊 Hyperliquid アカウント状況")
        print("="*70)
        
        # Perps
        state = self.get_user_state()
        if state:
            # マージン情報の取得 (Cross or Isolated)
            summary = state.get('crossMarginSummary', {}) or state.get('marginSummary', {})
            account_val = float(summary.get('accountValue', 0))
            usd_bal = float(summary.get('totalRawUsd', 0))
            
            print(f"💰 Perps (先物)")
            print(f"   アカウント価値: ${account_val:.2f}")
            print(f"   USDC残高: ${usd_bal:.2f}")
            
            # ポジション
            positions = state.get('assetPositions', [])
            has_pos = False
            if positions:
                for pos in positions:
                    p = pos.get('position', {})
                    szi = float(p.get('szi', 0))
                    if szi != 0:
                        if not has_pos:
                            print("\n   📈 オープンポジション:")
                            has_pos = True
                        
                        coin = p.get('coin')
                        side = "LONG" if szi > 0 else "SHORT"
                        print(f"      {coin}: {side} {abs(szi)} @ ${float(p.get('entryPx', 0)):.2f} (PnL: ${float(p.get('unrealizedPnl', 0)):+.2f})")
            
            if not has_pos:
                print("   ℹ️ オープンポジションなし")
        
        print("="*70 + "\n")

# =========================================================================
# CLI エントリーポイント
# =========================================================================
def main():
    import sys
    trader = HyperliquidSDKTrader()
    
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python hyperliquid_sdk_trader.py status")
        print("  python hyperliquid_sdk_trader.py price ETH")
        print("  python hyperliquid_sdk_trader.py buy ETH 0.01")
        print("  python hyperliquid_sdk_trader.py sell ETH 0.01")
        print("  python hyperliquid_sdk_trader.py close ETH")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "status":
        trader.print_account_status()
    elif cmd == "price":
        sym = sys.argv[2] if len(sys.argv) > 2 else "ETH"
        p = trader.get_current_price(sym)
        if p: print(f"💰 {sym}: ${p:.2f}")
    elif cmd in ["buy", "sell"]:
        if len(sys.argv) < 4:
            print(f"❌ 使用方法: python hyperliquid_sdk_trader.py {cmd} ETH 0.01")
            return
        sym = sys.argv[2]
        sz = float(sys.argv[3])
        trader.place_order(sym, is_buy=(cmd=="buy"), size=sz, order_type="market")
    elif cmd == "close":
        sym = sys.argv[2] if len(sys.argv) > 2 else "ETH"
        trader.close_position(sym)
    else:
        print(f"❌ 不明なコマンド: {cmd}")

if __name__ == "__main__":
    main()