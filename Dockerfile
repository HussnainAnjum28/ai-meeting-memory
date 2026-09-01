FROM python:3.12-slim

# Install ffmpeg (required for audio processing and diarization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Cloud Run provides the PORT environment variable; default to 8080 locally
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app.api_server:app --host 0.0.0.0 --port ${PORT}
