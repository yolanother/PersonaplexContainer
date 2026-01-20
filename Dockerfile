FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir moshi

# Copy custom server script
COPY server.py .

# Expose WebSocket port
EXPOSE 8080

# Run the custom server
CMD ["python", "server.py"]
