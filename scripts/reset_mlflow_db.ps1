# Reinitialise la base MLflow pour corriger l'erreur "Can't locate revision d3e4f5a6b7c8"
# Usage: .\scripts\reset_mlflow_db.ps1
# Pensez a arreter Streamlit et tout processus utilisant mlflow.db avant.

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $root "pyproject.toml"))) { $root = (Get-Location).Path }
$dbPath = Join-Path $root "mlflow.db"
$bakPath = Join-Path $root "mlflow.db.bak"

if (Test-Path $dbPath) {
    Copy-Item $dbPath $bakPath -Force
    Write-Host "Sauvegarde: $bakPath"
    Remove-Item $dbPath -Force
    Write-Host "Suppression de mlflow.db"
}
Write-Host "Une nouvelle base sera creee au prochain lancement de MLflow (mlflow ui ou run_app.py --mlflow)."
$uriPath = ($root -replace '\\', '/') + '/mlflow.db'
Write-Host "Pour lancer l'UI: uv run mlflow ui --port 5000 --backend-store-uri sqlite:///$uriPath"
