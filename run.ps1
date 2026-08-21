<#
    ClaimCompanion launcher.

        .\run.ps1            start the API and the UI
        .\run.ps1 -Seed      wipe and regenerate the synthetic dataset first
        .\run.ps1 -Stop      stop both services
        .\run.ps1 -Evals     run the CI gates instead of starting anything
        .\run.ps1 -NoLlm     start in template mode (the degradation demo)

    Port 8010 is used for the API because proxy.py in the parent folder holds 8000.
#>
param(
    [switch]$Seed,
    [switch]$Stop,
    [switch]$Evals,
    [switch]$NoLlm,
    [int]$ApiPort = 8010,
    [int]$UiPort  = 8501
)

$ErrorActionPreference = "Stop"
$Root   = $PSScriptRoot
$Python = Join-Path $Root "..\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Can't find the virtualenv at $Python" -ForegroundColor Red
    exit 1
}

$env:PYTHONPATH = "$Root\backend;$Root"
$env:CLAIMCOMPANION_API = "http://127.0.0.1:$ApiPort/api/v1"
if ($NoLlm) { $env:LLM_ENABLED = "false" } else { $env:LLM_ENABLED = "true" }

function Stop-Services {
    $killed = 0
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*uvicorn*app.main*" -or
                       $_.CommandLine -like "*streamlit*frontend/app.py*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $killed++ }
    Write-Host "Stopped $killed process(es)." -ForegroundColor Yellow
}

function Wait-For($url, $name, $seconds = 45) {
    for ($i = 0; $i -lt $seconds; $i++) {
        try {
            $probe = & $Python -c "import requests,sys;s=requests.Session();s.trust_env=False;sys.exit(0 if s.get('$url',timeout=3).ok else 1)"
            if ($LASTEXITCODE -eq 0) { Write-Host "  $name is up." -ForegroundColor Green; return $true }
        } catch { }
        Start-Sleep -Seconds 1
    }
    Write-Host "  $name did not come up in ${seconds}s." -ForegroundColor Red
    return $false
}

# ---------------------------------------------------------------- modes
if ($Stop)  { Stop-Services; exit 0 }

if ($Evals) {
    Write-Host "`nRunning CI gates..." -ForegroundColor Cyan
    & $Python -m evals.run_evals --skip-documents
    exit $LASTEXITCODE
}

Stop-Services
Start-Sleep -Seconds 2

if ($Seed) {
    Write-Host "`nGenerating the synthetic dataset (seed 42)..." -ForegroundColor Cyan
    & $Python -m datagen.generate --seed 42 --customers 20 --claims-per 2
}

# ---------------------------------------------------------------- start
Write-Host "`nStarting the API on :$ApiPort ..." -ForegroundColor Cyan
Start-Process -FilePath $Python `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$ApiPort" `
    -WorkingDirectory $Root -NoNewWindow `
    -RedirectStandardOutput "$env:TEMP\cc_api.log" -RedirectStandardError "$env:TEMP\cc_api_err.log"

if (-not (Wait-For "http://127.0.0.1:$ApiPort/health" "API")) {
    Write-Host "`nLast lines of $env:TEMP\cc_api_err.log:" -ForegroundColor Red
    Get-Content "$env:TEMP\cc_api_err.log" -Tail 20
    exit 1
}

Write-Host "`nStarting the UI on :$UiPort ..." -ForegroundColor Cyan
Start-Process -FilePath $Python `
    -ArgumentList "-m", "streamlit", "run", "frontend/app.py",
                  "--server.port", "$UiPort", "--server.headless", "true" `
    -WorkingDirectory $Root -NoNewWindow `
    -RedirectStandardOutput "$env:TEMP\cc_ui.log" -RedirectStandardError "$env:TEMP\cc_ui_err.log"

Wait-For "http://127.0.0.1:$UiPort/_stcore/health" "UI" | Out-Null

# ---------------------------------------------------------------- summary
Write-Host ""
Write-Host "  ClaimCompanion is running" -ForegroundColor Green
Write-Host "  ------------------------------------------------------------"
Write-Host "  Portal      http://localhost:$UiPort"
Write-Host "  API docs    http://127.0.0.1:$ApiPort/docs"
if ($NoLlm) {
    Write-Host "  LLM         DISABLED - template mode" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "  Customer    priya@example.com      code 000000"
Write-Host "  Reviewer    agent.marcus           code 000000   (Staff tab)"
Write-Host "  Manager     manager.elena          code 000000   (Staff tab)"
Write-Host ""
Write-Host "  Logs        $env:TEMP\cc_api_err.log / cc_ui_err.log"
Write-Host "  Stop        .\run.ps1 -Stop"
Write-Host ""
