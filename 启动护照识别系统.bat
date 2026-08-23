@echo off
cd /d "D:\passport reader"

set "PYTHON=python"
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"

"%PYTHON%" "scripts\start_web_app.py"
if errorlevel 1 pause
