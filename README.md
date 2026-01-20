# Personaplex (Moshi) Server

This directory contains the server-side integration for the [NVIDIA PersonaPlex-7B-v1](https://huggingface.co/nvidia/personaplex-7b-v1) model.

It uses Docker to run the official [Moshi](https://github.com/kyutai-labs/moshi) server in an isolated environment.

## Prerequisites
- **NVIDIA GPU**: Required for inference (7B parameters).
- **Docker Desktop**: With WSL2 backend and NVIDIA Container Toolkit support.
- **Hugging Face Account**: You need an access token to download the model weights.

## Quick Start

### 1. Configuration
The server is configured via a `.env` file.
Run the startup script once to generate this file, or create it manually from `.env.example`.

**Important**: You must add your Hugging Face token to the `.env` file.
```properties
HF_TOKEN=hf_...
```

### 2. Run
Use the provided scripts to start the server:

**Windows (PowerShell)**:
```powershell
.\start.ps1
```

**Linux / Mac / WSL (Bash)**:
```bash
./start.sh
```

The server will be available at: `ws://localhost:8080/api/chat` (or the port defined in `.env`).

---

## Manual Setup

If you prefer to run `docker compose` directly:

1.  Create `.env` file:
    ```bash
    cp .env.example .env
    # Edit .env and set HF_TOKEN
    ```
2.  Run Docker Compose:
    ```bash
    docker compose up --build
    ```

## API Endpoint
The service exposes a WebSocket endpoint:
- **URL**: `ws://localhost:8080/api/chat`
- **Protocol**: Binary WebSocket (Opus Audio Frames + Text Tokens).

## Data Persistence
Model weights are downloaded to the `./data` directory (mounted to `/root/.cache/huggingface` in the container) to avoid re-downloading them on every restart.
