FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# 'moshi' is the official package from Kyutai Labs
# We install it directly from PyPI.
RUN pip install --no-cache-dir moshi

# Expose WebSocket port
EXPOSE 8080

# Run the server using the official module entry point
# We use nvidia/personaplex-7b-v1 as the repo
CMD ["python", "-m", "moshi.server", "--host", "0.0.0.0", "--port", "8080", "--hf-repo", "nvidia/personaplex-7b-v1"]
