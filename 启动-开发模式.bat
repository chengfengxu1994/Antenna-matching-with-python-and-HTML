@echo off
chcp 65001 >nul
cd /d "%~dp0"
python start.py --dev
if errorlevel 1 pause
