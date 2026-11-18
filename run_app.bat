@echo off
chcp 65001 > nul
echo ============================================================
echo   👁️ KHỞI ĐỘNG AI VISION ASSISTANT (GRADIO WEB APP)
echo ============================================================
echo Đang mở trình duyệt tại: http://127.0.0.1:7860 ...
start "" "http://127.0.0.1:7860"
echo Đang khởi chạy web server...
python app/main.py
pause
