# google_sheets_logger.py (時間軸カラム追加版)

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
    「AI分析」シートに時間軸(Timeframe)カラムを追加
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
        self.flush_interval = 60  # 1分
        
        self._authenticate()
        self._setup_spreadsheet()

    def _authenticate(self):
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
        try:
            self.spreadsheet = self.client.open(self.spreadsheet_name)
            print(f"📊 既存シート '{self.spreadsheet_name}' を開きました")
        except gspread.SpreadsheetNotFound:
            print(f"🆕 新規シート '{self.spreadsheet_name}' を作成します...")
            self.spreadsheet = self.client.create(self.spreadsheet_name)
            try:
                self.spreadsheet.share(self.client.auth.service_account_email, perm_type='user', role='owner')
            except: pass
            print(f"✅ 作成完了: {self.spreadsheet.url}")

        self._ensure_sheets_exist()

    def _ensure_sheets_exist(self):
        """必要なシートを作成・ヘッダー設定"""
        
        # AI分析シートの2列目に「時間軸」を追加
        sheets_config = [
            ("実行履歴", ["日時", "アクション", "方向", "数量(ETH)", "価格($)", "手数料($)", "実現損益($)", "残高($)", "理由"]),
            ("AI分析", ["日時", "時間軸", "現在価格", "AI判断", "信頼度(%)", "上昇確率(%)", "下降確率(%)", "市場レジーム", "使用モデル", "RSI", "Volatility", "前回答え合わせ", "予測判定"]),
            ("資産推移", ["日時", "総資産($)", "利用可能($)", "ポジション価値($)", "未実現損益($)", "累積実現損益($)"]),
            ("Trade_History", ["Exit Time", "Symbol", "Side", "Size", "Entry Price", "Exit Price", "PnL ($)", "Result", "Duration", "Entry Reason", "Exit Reason"])
        ]

        for title, headers in sheets_config:
            self._setup_sheet(title, headers)

    def _setup_sheet(self, title: str, headers: List[str]):
        try:
            sheet = self.spreadsheet.worksheet(title)
            # 既存シートのヘッダー更新（カラムが増えた場合の対応）
            current_headers = sheet.row_values(1)
            if len(current_headers) < len(headers):
                print(f"⚠️ シート '{title}' のヘッダーを更新します...")
                sheet.resize(cols=len(headers))
                # 1行目を上書き
                for i, h in enumerate(headers):
                    sheet.update_cell(1, i+1, h)
                    
        except gspread.WorksheetNotFound:
            sheet = self.spreadsheet.add_worksheet(title=title, rows=1000, cols=len(headers))
            sheet.append_row(headers)
            sheet.format('A1:Z1', {
                "backgroundColor": {"red": 0.2, "green": 0.2, "blue": 0.2},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                "horizontalAlignment": "CENTER"
            })
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
        """AI分析に追加 (時間軸対応)"""
        up_prob = data.get('up_prob', 0) * 100
        down_prob = data.get('down_prob', 0) * 100
        
        timeframe = data.get('timeframe', '-') 
        
        row = [
            data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            timeframe,
            data.get('price'),
            data.get('action'),
            data.get('confidence'),
            f"{up_prob:.1f}",
            f"{down_prob:.1f}",
            data.get('market_regime', 'UNKNOWN'),
            data.get('model_used', 'ENSEMBLE'),
            f"{data.get('rsi', 0):.1f}",
            f"{data.get('volatility', 0):.2f}",
            data.get('eval_result', '-'),
            data.get('prediction_result', '-')
        ]
        self.buffer['ai_analysis'].append(row)
        self._try_flush()

    def log_equity(self, data: Dict[str, Any]):
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
        if not self.spreadsheet: return
        try:
            pnl = float(data.get('pnl', 0))
            if pnl > 0: result_icon = "🏆 WIN"
            elif pnl < 0: result_icon = "💀 LOSE"
            else: result_icon = "⚪ DRAW"
            
            row = [
                str(data.get('exit_time')),
                data.get('symbol'),
                data.get('side'),
                data.get('size'),
                data.get('entry_price'),
                data.get('exit_price'),
                pnl,
                result_icon,
                str(data.get('duration')),
                data.get('entry_reason'),
                data.get('exit_reason')
            ]
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
        if self.buffer[buffer_key]:
            sheet = self.spreadsheet.worksheet(sheet_name)
            rows = list(self.buffer[buffer_key])
            rows.reverse()
            sheet.insert_rows(rows, row=2, value_input_option='USER_ENTERED')
            if buffer_key == 'ai_analysis':
                self._apply_ai_formatting(sheet, rows)
            self.buffer[buffer_key].clear()

    def _apply_ai_formatting(self, sheet, rows):
        """AI分析シートの条件付き色塗り"""
        try:
            formats = []
            for i, row_data in enumerate(rows):
                # headers: ["日時", "時間軸", "現在価格", "AI判断", ...] -> AI判断は Index 3
                action = row_data[3]
                
                color = None
                if action == 'BUY' or action == 'STRONG_BUY':
                    color = {"red": 0.85, "green": 0.95, "blue": 1.0}
                elif action == 'SELL' or action == 'STRONG_SELL':
                    color = {"red": 1.0, "green": 0.85, "blue": 0.85}
                elif action == 'CLOSE':
                    color = {"red": 1.0, "green": 1.0, "blue": 0.85}
                else:
                    color = {"red": 1.0, "green": 1.0, "blue": 1.0}
                
                if color:
                    # 範囲計算 (A列〜M列) ※列が増えたのでMまで
                    row_idx = 2 + i
                    rng = f"A{row_idx}:M{row_idx}"
                    formats.append({"range": rng, "format": {"backgroundColor": color}})
            
            if formats:
                sheet.batch_format(formats)
        except Exception as e:
            print(f"⚠️ シート色塗りエラー (無視して続行): {e}")

    def force_flush(self):
        try:
            self._flush_buffer_to_sheet("実行履歴", 'executions')
            self._flush_buffer_to_sheet("AI分析", 'ai_analysis')
            self._flush_buffer_to_sheet("資産推移", 'equity')
            self.last_flush = time.time()
            print("📝 Google Sheetsログ同期完了")
        except Exception as e:
            print(f"⚠️ ログ書き込みエラー: {e}")

    def get_spreadsheet_url(self) -> str:
        return self.spreadsheet.url if self.spreadsheet else "未接続"