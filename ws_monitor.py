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
        
        # 最新の板情報を保持する変数
        self.latest_book = {
            'bids': [], 
            'asks': [], 
            'timestamp': 0,
            'imbalance': 0.0 
        }
        
        # ★追加: 最新のOIを保持する変数
        self.latest_oi = 0.0
        self.oi_timestamp = 0
        
        self.lock = threading.Lock() 

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            channel = data.get('channel')
            
            # 1. 板情報 (l2Book)
            if channel == 'l2Book':
                raw_data = data.get('data', {})
                levels = raw_data.get('levels', [])
                
                if len(levels) == 2:
                    bids = levels[0] 
                    asks = levels[1] 
                    current_time = time.time()
                    
                    # インバランス計算
                    bid_vol = sum([float(b['sz']) for b in bids[:5]])
                    ask_vol = sum([float(a['sz']) for a in asks[:5]])
                    total_vol = bid_vol + ask_vol
                    
                    imbalance = 0.0
                    if total_vol > 0:
                        imbalance = (bid_vol - ask_vol) / total_vol
                    
                    with self.lock:
                        self.latest_book['bids'] = bids
                        self.latest_book['asks'] = asks
                        self.latest_book['timestamp'] = current_time
                        self.latest_book['imbalance'] = imbalance

            # 2. ★追加: 資産コンテキスト (activeAssetCtx) からOIを取得
            elif channel == 'activeAssetCtx':
                raw_data = data.get('data', {})
                ctx = raw_data.get('ctx', {})
                
                # openInterest を取得 (文字列で来る場合があるのでfloat変換)
                oi_str = ctx.get('openInterest', '0')
                try:
                    oi_val = float(oi_str)
                except:
                    oi_val = 0.0
                
                with self.lock:
                    self.latest_oi = oi_val
                    self.oi_timestamp = time.time()
                    
        except Exception as e:
            print(f"⚠️ WS Parse Error: {e}")

    def _on_error(self, ws, error):
        print(f"⚠️ WS Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print("🔌 WS Disconnected")

    def _on_open(self, ws):
        print(f"⚡ WS Connected: Subscribing to L2Book & ActiveAssetCtx for {self.symbol}")
        
        # 購読メッセージ送信 (板情報 + 資産情報)
        subscribe_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": self.symbol
            }
        }
        ws.send(json.dumps(subscribe_msg))
        
        # ★追加購読: OIを含む詳細情報の取得
        oi_msg = {
            "method": "subscribe",
            "subscription": {
                "type": "activeAssetCtx",
                "coin": self.symbol
            }
        }
        ws.send(json.dumps(oi_msg))

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
        self.running = True
        websocket.enableTrace(False)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()



    def _run_loop(self):
        """接続が切れても再接続し続けるループ"""
        while self.running:
            try:
                print(f"⚡ WS Connecting to {self.ws_url}...")
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                # 接続が切れるまでブロック
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print(f"⚠️ WS Connection failed: {e}")
            
            if self.running:
                print("⏳ Reconnecting in 5 seconds...")
                time.sleep(5)

                

    def get_latest_imbalance(self):
        with self.lock:
            if time.time() - self.latest_book['timestamp'] > 5.0: # 許容時間を少し緩和
                return 0.0
            return self.latest_book['imbalance']
            
    def get_latest_oi(self):
        """★追加: 最新のOIを取得"""
        with self.lock:
            # データが古すぎる(60秒以上更新なし)場合は警告扱いだが、
            # OIは頻繁に変わらないので、前回の値を信用して返す
            return self.latest_oi