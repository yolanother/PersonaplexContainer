# Ensure we are in the script's directory
Set-Location $PSScriptRoot

if (-not (Test-Path .env)) {
    Write-Host "⚠️  .env file not found." -ForegroundColor Yellow
    Write-Host "Creating .env from .env.example..."
    Copy-Item .env.example .env
    Write-Host "✅ .env created." -ForegroundColor Green
    Write-Host ""
    Write-Host "❗ ACTION REQUIRED: Please open the '.env' file and add your Hugging Face token (HF_TOKEN)." -ForegroundColor Red
    Write-Host "The model 'nvidia/personaplex-7b-v1' requires authentication."
    exit
}

Write-Host "🚀 Starting Personaplex Server..." -ForegroundColor Cyan
docker compose up --build
