#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "⚠️  .env file not found."
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env created."
    echo ""
    echo "❗ ACTION REQUIRED: Please open the '.env' file and add your Hugging Face token (HF_TOKEN)."
    echo "The model 'nvidia/personaplex-7b-v1' requires authentication."
    exit 1
fi

echo "🚀 Starting Personaplex Server..."
docker compose up --build
