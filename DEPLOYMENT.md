# Deployment Guide - Deepfake Detection API

## Fixed OpenCV libGL.so.1 Error

The error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` has been resolved with the following fixes:

### 1. Updated Requirements
- Using `opencv-python-headless==4.8.1.78` (headless version for deployment)
- Pinned all dependency versions for stability

### 2. Environment Variables
Added headless operation environment variables:
```bash
OPENCV_OPENCL_RUNTIME=""
QT_QPA_PLATFORM=offscreen
DISPLAY=:99
```

### 3. Dockerfile System Dependencies
The Dockerfile now includes all necessary system libraries:
- `libglib2.0-0`, `libsm6`, `libxext6` - Core OpenCV dependencies
- `libgstreamer*` packages - MediaPipe dependencies
- `libhdf5-dev` - TensorFlow support

## Quick Deployment

### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```

### Using Docker directly
```bash
docker build -t deepfake-api .
docker run -p 8000:8000 \
  -e OPENCV_OPENCL_RUNTIME="" \
  -e QT_QPA_PLATFORM=offscreen \
  -e DISPLAY=:99 \
  deepfake-api
```

### Manual Python Setup (Local development)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENCV_OPENCL_RUNTIME=""
export QT_QPA_PLATFORM=offscreen
export DISPLAY=:99

# Run the application
python main.py
```

## Platform-Specific Deployment

### Railway
1. Push to GitHub
2. Connect to Railway
3. Environment variables are already set in main.py
4. Railway will use the PORT environment variable

### Heroku
```bash
# Add buildpack for system dependencies
heroku buildpacks:add --index 1 heroku-community/apt
```

Create `Aptfile`:
```
libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
libgstreamer1.0-0
libgstreamer-plugins-base1.0-0
```

### DigitalOcean App Platform
The Dockerfile and environment variables will work out of the box.

### AWS/GCP/Azure Container Services
Use the provided Dockerfile with the container service of your choice.

## Health Check
The API includes a health check endpoint at `/health`:
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### If you still get libGL.so.1 errors:
1. Ensure you're using `opencv-python-headless` (not `opencv-python`)
2. Check environment variables are set before importing cv2
3. In Dockerfile, add additional dependencies if needed:
   ```dockerfile
   RUN apt-get update && apt-get install -y libgl1-mesa-glx
   ```

### MediaPipe issues:
If MediaPipe fails, the app falls back to Haar Cascade detection automatically.

### Memory issues:
Adjust batch size in the code or add memory limits to Docker:
```bash
docker run --memory=4g deepfake-api
```

## File Structure
```
BitnBuild/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container definition
├── docker-compose.yml  # Docker Compose setup
├── DEPLOYMENT.md       # This file
├── models/             # Place your model files here
│   ├── Meso4_DF.h5
│   └── mesonet_idx_to_class.pkl
├── uploads/            # Temporary upload storage
└── results/            # Processing results
```

## Environment Variables
- `MODEL_PATH`: Path to the MesoNet model file (default: models/Meso4_DF.h5)
- `LABEL_PATH`: Path to the label mapping file (default: models/mesonet_idx_to_class.pkl)
- `PORT`: Port to run on (default: 8000)
- `OPENCV_OPENCL_RUNTIME`: Set to empty string for headless operation
- `QT_QPA_PLATFORM`: Set to "offscreen" for headless operation
- `DISPLAY`: Set to ":99" for headless operation
