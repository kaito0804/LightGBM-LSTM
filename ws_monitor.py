import websocket
import threading
import json
import time

class OrderBookMonitor:
    def __init__(self, symbol='ETH'):
        self.symbol = symbol.upper()
        self.ws_url = "wss://api.hyperliquid.xyz/ws"
        self.ws = None
        self.thread = None
        self.running = False
        
        # 最新の板情報を保持する変数 (ここに常に最新データが入る)
        self.latest_book = {
            'bids': [], # 買い板 [[price, size], ...]
            'asks': [], # 売り板
            'timestamp': 0,
            'imbalance': 0.0 # 買い圧/売り圧の指標
        }
        self.lock = threading.Lock() # データ競合を防ぐ鍵

    def _on_message(self, ws, message):
        data = json.loads(message)
        
        # 板情報(l2Book)の更新を受け取る
        if data.get('channel') == 'l2Book':
            raw_data = data.get('data', {})
            levels = raw_data.get('levels', [])
            
            if len(levels) == 2:
                bids = levels[0] # 買い板
                asks = levels[1] # 売り板
                
                # 計算処理
                current_time = time.time()
                
                # インバランスの計算 (上位5本の板厚で判定)
                bid_vol = sum([float(b['sz']) for b in bids[:5]])
                ask_vol = sum([float(a['sz']) for a in asks[:5]])
                total_vol = bid_vol + ask_vol
                
                imbalance = 0.0
                if total_vol > 0:
                    imbalance = (bid_vol - ask_vol) / total_vol
                
                # データを安全に更新
                with self.lock:
                    self.latest_book['bids'] = bids
                    self.latest_book['asks'] = asks
                    self.latest_book['timestamp'] = current_time
                    self.latest_book['imbalance'] = imbalance

    def _on_error(self, ws, error):
        print(f"⚠️ WS Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print("🔌 WS Disconnected")

    def _on_open(self, ws):
        print("⚡ WS Connected: Subscribing to L2Book")
        # 購読メッセージ送信
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": self.symbol
            }
        }
        ws.send(json.dumps(subscribe_msg))

    def start(self):
        """監視をバックグラウンドで開始"""
        self.running = True
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        # スレッド（別働隊）として起動
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.daemon = True # メインが終了したら一緒に死ぬ設定
        self.thread.start()

    def get_latest_imbalance(self):
        """メインプログラムから呼び出す用"""
        with self.lock:
            # データが古すぎる(3秒以上前)場合は信頼しない
            if time.time() - self.latest_book['timestamp'] > 3.0:
                return 0.0
            return self.latest_book['imbalance']

    def get_best_prices(self):
        """最良気配値を取得"""
        with self.lock:
            if not self.latest_book['bids'] or not self.latest_book['asks']:
                return None, None
            best_bid = float(self.latest_book['bids'][0]['px'])
            best_ask = float(self.latest_book['asks'][0]['px'])
            return best_bid, best_ask