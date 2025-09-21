# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set environment variables for headless operation
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DISPLAY=:99
ENV QT_QPA_PLATFORM=offscreen
ENV OPENCV_OPENCL_RUNTIME=""

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    # Essential system packages
    gcc \
    g++ \
    pkg-config \
    # OpenCV dependencies (headless)
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-glog0v5 \
    # MediaPipe dependencies
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-good1.0-0 \
    libgstreamer-plugins-bad1.0-0 \
    # For TensorFlow
    libhdf5-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
