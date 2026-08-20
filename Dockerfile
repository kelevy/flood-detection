# Base image: a slim Linux image with Python 3.10 pre-installed
FROM python:3.10-slim

# System dependencies needed by rasterio (GDAL) and OpenCV (used internally
# by segmentation_models_pytorch/albumentations)
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (separate layer from code copy below).
# Docker caches each step - if only your code changes (not dependencies),
# this step is skipped on rebuild, making iteration much faster.
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application code
COPY src/ ./src/
COPY api/ ./api/

# Copy the trained model checkpoint into the image.
# For a real production system you'd typically load this from GCS at
# startup instead of baking it into the image - simpler here since the
# checkpoint is small and doesn't change often.
COPY models/best_model.pt ./models/best_model.pt

WORKDIR /app/api

# Cloud Run injects the PORT environment variable; default to 8080 locally
ENV PORT=8080
ENV MODEL_CHECKPOINT=/app/models/best_model.pt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
