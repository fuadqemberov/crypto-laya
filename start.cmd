@echo off
cd /d "%~dp0"
if exist .native\freqtrade\Scripts\python.exe set "FREQTRADE_PYTHON=%CD%\.native\freqtrade\Scripts\python.exe"
if exist .native\jesse\Scripts\python.exe set "JESSE_PYTHON=%CD%\.native\jesse\Scripts\python.exe"
.venv\Scripts\python.exe app.py
pause
