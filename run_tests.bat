@echo off
chcp 65001 > nul
echo ============================================================
echo   🧪 CHẠY BỘ KIỂM THỬ TỰ ĐỘNG (UNIT TESTS)
echo ============================================================
python -m pytest tests/ -v
pause
