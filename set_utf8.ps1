<#
.SYNOPSIS
    设置当前 PowerShell 会话为 UTF-8 编码，解决中文乱码问题。
.DESCRIPTION
    - 控制台输出编码 → UTF-8（[Console]::OutputEncoding）
    - 管道输出编码 → UTF-8（$OutputEncoding）
    - Get-Content / Out-File 默认编码 → UTF-8
    - Python 的 stdin/stdout 编码 → UTF-8（$env:PYTHONUTF8=1）
.EXAMPLE
    # 在当前会话生效（推荐）
    . .\set_utf8.ps1

    # 或直接运行（部分设置仅对子进程生效）
    .\set_utf8.ps1
#>

Write-Host "正在设置 UTF-8 编码环境..." -ForegroundColor Cyan

# ── 控制台编码 ──────────────────────────────────────────────
# 让 PowerShell 控制台能正确显示 UTF-8 中文
[Console]::OutputEncoding = [Text.Encoding]::UTF8

# ── PowerShell 管道/重定向编码 ──────────────────────────────
$OutputEncoding = [Text.Encoding]::UTF8

# ── Get-Content / Out-File 默认编码 ─────────────────────────
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# ── Python 的 IO 编码 ────────────────────────────────────────
# Python 3.7+ 支持 PYTHONUTF8 环境变量：使 open() 默认使用 UTF-8
$env:PYTHONUTF8 = '1'

Write-Host "✅ 已完成！当前控制台编码: $([Console]::OutputEncoding.WebName)" -ForegroundColor Green
Write-Host ""
Write-Host "使用提示:" -ForegroundColor Yellow
Write-Host "  • 读取文件: Get-Content -Encoding UTF8 .\文件.txt" -ForegroundColor Gray
Write-Host "  • 读取文件(自动检测BOM): Get-Content .\文件.txt" -ForegroundColor Gray
Write-Host "  • 运行 Python: python start.py" -ForegroundColor Gray
Write-Host "  • 如需持久化，将此脚本加入 PowerShell profile:" -ForegroundColor Gray
Write-Host "    echo `". '$PWD\set_utf8.ps1'`" >> `$PROFILE" -ForegroundColor Gray
