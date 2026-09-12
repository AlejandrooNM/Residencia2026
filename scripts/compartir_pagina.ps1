# Comparte CardioIA con un enlace publico temporal (Cloudflare Tunnel).
# Uso (PowerShell):
#   cd E:\Residencia2026\Proyecto
#   .\scripts\compartir_pagina.ps1
#
# Deja la ventana abierta mientras el asesor navega.

$ErrorActionPreference = "Stop"
$raiz = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $raiz

$python = "E:\Residencia2026\venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$tools = "E:\Residencia2026\tools"
$cloudflared = Join-Path $tools "cloudflared.exe"
New-Item -ItemType Directory -Force -Path $tools | Out-Null

if (-not (Test-Path $cloudflared)) {
  Write-Host "Descargando cloudflared (solo la primera vez)..." -ForegroundColor Cyan
  $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
  Invoke-WebRequest -Uri $url -OutFile $cloudflared
}

Write-Host "Iniciando API en modo compartir..." -ForegroundColor Cyan
$api = Start-Process -FilePath $python `
  -ArgumentList @("scripts/iniciar_api.py", "--compartir") `
  -WorkingDirectory $raiz.Path `
  -PassThru

Start-Sleep -Seconds 5

Write-Host ""
Write-Host "Creando enlace publico temporal..." -ForegroundColor Cyan
Write-Host "Copia la URL https://....trycloudflare.com y enviasela a tu asesor." -ForegroundColor Yellow
Write-Host "NO cierres esta ventana mientras el asesor use la pagina." -ForegroundColor Yellow
Write-Host ""

try {
  & $cloudflared tunnel --url "http://127.0.0.1:8000"
}
finally {
  if ($api -and -not $api.HasExited) {
    Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
  }
}
