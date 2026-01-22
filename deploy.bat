@echo off
echo ---------------------------------------
echo ? VPSへのデプロイを開始します (models除外)
echo ---------------------------------------

:: 1. modelsフォルダを一時的に一つ上の階層へ退避（隠す）
if exist "models" (
    move models ..\models_temp_backup
    echo ? modelsフォルダを一時退避しました
) else (
    echo ?? modelsフォルダが見つかりません（スキップ）
)

:: 2. SCPコマンド実行
echo ? ファイルを送信中...
scp -i C:\Users\USER\.ssh\trade-bot.pem -r "C:\Users\USER\Desktop\自分のサイト\Python\trade_bot\LightGBM_LSTM\*" root@162.43.75.239:/opt/botter/LightGBM_LSTM/

:: 3. modelsフォルダを元の場所に戻す
if exist "..\models_temp_backup" (
    move ..\models_temp_backup models
    echo ? modelsフォルダを元に戻しました
)

echo ---------------------------------------
echo ? デプロイ完了！
echo ---------------------------------------
pause