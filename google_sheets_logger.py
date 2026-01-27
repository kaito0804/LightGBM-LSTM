# google_sheets_logger.py
# Google Sheetsへの取引ログ記録・可視化システム（リニューアル版・降順記録・色塗り対応）

import os
import time
from datetime import datetime
from typing import Dict, List, Any
from collections import deque
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv()

class GoogleSheetsLogger:
    """
    Google Sheetsへ取引結果を記録・可視化
    分析しやすいように「実行」「AI思考」「資産」にシートを分離
    ★新しいログを上部（ヘッダー直下）に追加する仕様
    """
    
    DEFAULT_SPREADSHEET_NAME = "Hyperliquid_AI_Journal"
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self, spreadsheet_name: str = None):
        self.spreadsheet_name = spreadsheet_name or self.DEFAULT_SPREADSHEET_NAME
        self.client = None
        self.spreadsheet = None
        self.creds_path = os.getenv('GOOGLE_SHEETS_CREDENTIALS', 'credentials.json')
        
        # バッファリング
        self.buffer = {
            'executions': deque(maxlen=20),
            'ai_analysis': deque(maxlen=50),
            'equity': deque(maxlen=50)
        }
        self.last_flush = time.time()
        self.flush_interval = 60  # 5分
        
        self._authenticate()
        self._setup_spreadsheet()
    


    def _authenticate(self):
        """認証処理"""
        try:
            creds = Credentials.from_service_account_file(
                self.creds_path, scopes=self.SCOPES
            )
            self.client = gspread.authorize(creds)
            print("✅ Google Sheets認証成功")
        except Exception as e:
            print(f"❌ Google Sheets認証エラー: {e}")
            raise



    def _setup_spreadsheet(self):
        """スプレッドシートの準備"""
        try:
            self.spreadsheet = self.client.open(self.spreadsheet_name)
            print(f"📊 既存シート '{self.spreadsheet_name}' を開きました")
        except gspread.SpreadsheetNotFound:
            print(f"🆕 新規シート '{self.spreadsheet_name}' を作成します...")
            self.spreadsheet = self.client.create(self.spreadsheet_name)
            # 共有設定（自分のメールアドレスに共有）
            try:
                self.spreadsheet.share(self.client.auth.service_account_email, perm_type='user', role='owner')
            except:
                pass # サービスアカウント自身の所有になる場合はスキップ
            print(f"✅ 作成完了: {self.spreadsheet.url}")

        self._ensure_sheets_exist()



    def _ensure_sheets_exist(self):
        """必要な3つのシートを作成・ヘッダー設定"""
        
        sheets_config = [
            ("実行履歴", ["日時", "アクション", "方向", "数量(ETH)", "価格($)", "手数料($)", "実現損益($)", "残高($)", "理由"]),
            ("AI分析", ["日時", "現在価格", "AI判断", "信頼度(%)", "上昇確率(%)", "下降確率(%)", "市場レジーム", "使用モデル", "RSI", "Volatility", "前回比($)", "予測判定"]),
            ("資産推移", ["日時", "総資産($)", "利用可能($)", "ポジション価値($)", "未実現損益($)", "累積実現損益($)"]),
            ("Trade_History", ["Exit Time", "Symbol", "Side", "Size", "Entry Price", "Exit Price", "PnL ($)", "Result", "Duration", "Entry Reason", "Exit Reason"])
        ]

        for title, headers in sheets_config:
            self._setup_sheet(title, headers)



    def _setup_sheet(self, title: str, headers: List[str]):
        """シートの作成とヘッダー設定"""
        try:
            sheet = self.spreadsheet.worksheet(title)
        except gspread.WorksheetNotFound:
            sheet = self.spreadsheet.add_worksheet(title=title, rows=1000, cols=len(headers))
            sheet.append_row(headers)
            # ヘッダー装飾
            sheet.format('A1:Z1', {
                "backgroundColor": {"red": 0.2, "green": 0.2, "blue": 0.2},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                "horizontalAlignment": "CENTER"
            })
            # 1行目を固定
            sheet.freeze(rows=1)



    # ========== ログ記録メソッド ==========
    def log_execution(self, data: Dict[str, Any]):
        """実行履歴に追加"""
        row = [
            data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            data.get('action'),
            data.get('side'),
            data.get('size'),
            data.get('price'),
            data.get('fee'),
            data.get('realized_pnl', 0),
            data.get('balance'),
            data.get('reasoning')
        ]
        self.buffer['executions'].append(row)
        self._try_flush()



    def log_ai_analysis(self, data: Dict[str, Any]):
        """AI分析に追加"""
        # 確率を%表記に変換
        up_prob = data.get('up_prob', 0) * 100
        down_prob = data.get('down_prob', 0) * 100
        
        row = [
            data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            data.get('price'),
            data.get('action'),
            data.get('confidence'),
            f"{up_prob:.1f}",
            f"{down_prob:.1f}",
            data.get('market_regime', 'UNKNOWN'),
            data.get('model_used', 'ENSEMBLE'),
            f"{data.get('rsi', 0):.1f}",
            f"{data.get('volatility', 0):.2f}",
            data.get('price_diff', '-'),
            data.get('prediction_result', '-')
        ]
        self.buffer['ai_analysis'].append(row)
        self._try_flush()



    def log_equity(self, data: Dict[str, Any]):
        """資産推移に追加"""
        row = [
            data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            data.get('account_value'),
            data.get('available_balance'),
            data.get('position_value', 0),
            data.get('unrealized_pnl', 0),
            data.get('realized_pnl_cumulative', 0)
        ]
        self.buffer['equity'].append(row)
        self._try_flush()


    def log_trade_result(self, data: Dict[str, Any]):
        """詳細なトレード結果を記録 (Trade_Historyシート用)"""
        # シートが開かれていなければ何もしない
        if not self.spreadsheet: return
        
        try:
            # 勝ち負けの判定アイコン設定
            pnl = float(data.get('pnl', 0))
            if pnl > 0:
                result_icon = "🏆 WIN"
            elif pnl < 0:
                result_icon = "💀 LOSE"
            else:
                result_icon = "⚪ DRAW"
            
            row = [
                str(data.get('exit_time')),      # Exit Time
                data.get('symbol'),              # Symbol
                data.get('side'),                # Side
                data.get('size'),                # Size
                data.get('entry_price'),         # Entry Price
                data.get('exit_price'),          # Exit Price
                pnl,                             # PnL ($)
                result_icon,                     # Result
                str(data.get('duration')),       # Duration
                data.get('entry_reason'),        # Entry Reason
                data.get('exit_reason')          # Exit Reason
            ]
            
            # Trade_Historyは頻度が低いのでバッファを使わず直接書き込む
            # insert_row(..., index=2) でヘッダーの直下に挿入（最新が上に来る）
            sheet = self.spreadsheet.worksheet("Trade_History")
            sheet.insert_row(row, index=2, value_input_option='USER_ENTERED')
            print(f"📝 トレード履歴記録完了: {result_icon} ${pnl}")
            
        except Exception as e:
            print(f"⚠️ トレード履歴ログエラー: {e}")
            


    # ========== バッファ処理 ==========
    def _try_flush(self, force: bool = False):
        elapsed = time.time() - self.last_flush
        is_full = (len(self.buffer['executions']) >= 5 or 
                   len(self.buffer['ai_analysis']) >= 20 or 
                   len(self.buffer['equity']) >= 20)
        
        if force or elapsed >= self.flush_interval or is_full:
            self.force_flush()



    def _flush_buffer_to_sheet(self, sheet_name: str, buffer_key: str):
        """指定されたバッファの内容をシートに書き込むヘルパーメソッド"""
        if self.buffer[buffer_key]:
            sheet = self.spreadsheet.worksheet(sheet_name)
            rows = list(self.buffer[buffer_key])
            rows.reverse() # 新しい順にする
            
            # 書き込み
            sheet.insert_rows(rows, row=2, value_input_option='USER_ENTERED')
            
            # === AI分析シートの場合の色塗り処理 (追加) ===
            if buffer_key == 'ai_analysis':
                self._apply_ai_formatting(sheet, rows)

            self.buffer[buffer_key].clear()
            


    def _apply_ai_formatting(self, sheet, rows):
        """AI分析シートの条件付き色塗り"""
        try:
            formats = []
            for i, row_data in enumerate(rows):
                # headers: ["日時", "現在価格", "AI判断", ...] -> Index 2
                action = row_data[2]
                
                # 色の初期化（対象外のアクションはNoneのまま）
                color = None
                
                if action == 'BUY' or action == 'STRONG_BUY':
                    # 薄い青 (Light Cyan)
                    color = {"red": 0.85, "green": 0.95, "blue": 1.0}
                elif action == 'SELL' or action == 'STRONG_SELL':
                    # 薄い赤 (Light Red)
                    color = {"red": 1.0, "green": 0.85, "blue": 0.85}
                elif action == 'CLOSE':
                    # 薄い黄色
                    color = {"red": 1.0, "green": 1.0, "blue": 0.85}
                
                # 色が設定された場合のみフォーマットを適用 (HOLD, WAIT等は無視)
                if color:
                    # 範囲計算 (A列〜L列)
                    # 挿入された行は row=2 から始まる
                    row_idx = 2 + i
                    rng = f"A{row_idx}:L{row_idx}"
                    
                    formats.append({
                        "range": rng,
                        "format": {
                            "backgroundColor": color
                        }
                    })
            
            if formats:
                # バッチ処理で色を一括適用
                sheet.batch_format(formats)
                
        except Exception as e:
            print(f"⚠️ シート色塗りエラー (無視して続行): {e}")



    def force_flush(self):
        """バッファを書き込み（新しい順に上に追加）"""
        try:
            self._flush_buffer_to_sheet("実行履歴", 'executions')
            self._flush_buffer_to_sheet("AI分析", 'ai_analysis')
            self._flush_buffer_to_sheet("資産推移", 'equity')
            
            self.last_flush = time.time()
            print("📝 Google Sheetsログ同期完了 (Top-Insert)")
            
        except Exception as e:
            print(f"⚠️ ログ書き込みエラー: {e}")



    def get_spreadsheet_url(self) -> str:
        return self.spreadsheet.url if self.spreadsheet else "未接続"



if __name__ == "__main__":
    # テスト
    logger = GoogleSheetsLogger()
    print(f"URL: {logger.get_spreadsheet_url()}")
    # テストデータ
    logger.log_ai_analysis({
        'timestamp': datetime.now(), 'price': 3000, 'action': 'BUY', 'confidence': 80,
        'up_prob': 0.8, 'down_prob': 0.2, 'market_regime': 'TREND', 'model_used': 'LGBM'
    })
    logger.log_ai_analysis({
        'timestamp': datetime.now(), 'price': 3005, 'action': 'HOLD', 'confidence': 20,
        'up_prob': 0.4, 'down_prob': 0.4, 'market_regime': 'RANGE', 'model_used': 'LGBM'
    })
    logger.force_flush()