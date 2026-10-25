# Image Processing Platform -- launcher for Windows PowerShell.
#
#   .\run.ps1            start everything (API serves the built UI)
#   .\run.ps1 dev        API + Vite dev server with hot reload, two windows
#   .\run.ps1 setup      create the venv and install all dependencies
#   .\run.ps1 build      rebuild the UI bundle only
#   .\run.ps1 test       run the backend test suite
#   .\run.ps1 bench      per-operator timings
#
# If PowerShell refuses to run this, it is the execution policy, not the
# script:  powershell -ExecutionPolicy Bypass -File .\run.ps1

param([string]$Task = "start", [int]$Port = 8000)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Require-Venv {
    if (-not (Test-Path $Py)) {
        Write-Host "No virtual environment found. Run '.\run.ps1 setup' first." -ForegroundColor Yellow
        exit 1
    }
}

function Require-Build {
    if (-not (Test-Path (Join-Path $Root "backend\app\static\index.html"))) {
        Write-Host "UI bundle missing; building it now..." -ForegroundColor Cyan
        Invoke-Build
    }
}

function Invoke-Setup {
    Write-Host "Creating the virtual environment..." -ForegroundColor Cyan
    python -m venv (Join-Path $Root ".venv")
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r (Join-Path $Root "backend\requirements.txt")
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location (Join-Path $Root "frontend")
    npm install
    Pop-Location
    Invoke-Build
    Write-Host "Setup complete. Start it with: .\run.ps1" -ForegroundColor Green
}

function Invoke-Build {
    Push-Location (Join-Path $Root "frontend")
    npm run build
    Pop-Location
}

function Invoke-Start {
    Require-Venv
    Require-Build
    Write-Host ""
    Write-Host "  Image Processing Platform" -ForegroundColor Green
    Write-Host "  http://127.0.0.1:$Port        the app" -ForegroundColor Green
    Write-Host "  http://127.0.0.1:$Port/docs   API reference" -ForegroundColor DarkGray
    Write-Host "  Ctrl+C to stop" -ForegroundColor DarkGray
    Write-Host ""
    Push-Location (Join-Path $Root "backend")
    try { & $Py -m uvicorn app.main:app --port $Port } finally { Pop-Location }
}

function Invoke-Dev {
    Require-Venv
    $api = Join-Path $Root "backend"
    # --reload watches the backend; Vite handles the frontend in the other window.
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "Set-Location '$api'; & '$Py' -m uvicorn app.main:app --reload --port $Port"
    )
    Write-Host "API starting on http://127.0.0.1:$Port (separate window)" -ForegroundColor Cyan
    Write-Host "Opening the Vite dev server; use http://localhost:5173" -ForegroundColor Green
    Push-Location (Join-Path $Root "frontend")
    try { npm run dev } finally { Pop-Location }
}

switch ($Task.ToLower()) {
    "setup" { Invoke-Setup }
    "build" { Invoke-Build }
    "start" { Invoke-Start }
    "dev"   { Invoke-Dev }
    "test"  {
        Require-Venv
        Push-Location (Join-Path $Root "backend")
        try { & $Py -m pytest tests } finally { Pop-Location }
    }
    "bench" {
        Require-Venv
        Push-Location (Join-Path $Root "backend")
        try { & $Py -m tests.bench_ops } finally { Pop-Location }
    }
    default {
        Write-Host "Unknown task '$Task'. Use: setup, build, start, dev, test, bench"
        exit 1
    }
}
