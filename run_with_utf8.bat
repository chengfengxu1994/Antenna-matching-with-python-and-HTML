@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title RF Matching Tool (UTF-8)

echo ============================================
echo   RF Matching Tool
echo   已设置 UTF-8 编码 (代码页 65001)
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请安装 Python 3.7+ 并加入 PATH
    pause
    exit /b 1
)

:: 启动项目
python start.py %*

if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序异常退出，错误码: %errorlevel%
    pause
)
