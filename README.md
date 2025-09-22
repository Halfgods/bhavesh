# 🔍 Deepfake Detection API through image or Video 

Welcome to our cutting-edge deepfake detection service! 🚀

This project leverages the power of MesoNet architecture to unmask manipulated videos in real-time. Whether you're building a content verification platform or just curious about video authenticity, our API has got you covered! 🛡️

## 🎯 What are Deepfakes?

Deepfakes are synthetic media where a person's image or video is replaced with someone else's likeness using artificial intelligence. While this technology has creative applications, it can also be misused to spread misinformation. That's where our detection system comes in!

## 🧠 Our Approach: MesoNet Architecture

We use the powerful MesoNet model, a specialized convolutional neural network designed specifically for deepfake detection. Here's what makes it special:

### Model Architecture

```
Input Video → Face Detection → MesoNet Analysis → Authenticity Score
```

### Key Features of MesoNet:
- Designed to detect mesoscopic properties of images
- Focuses on the subtle artifacts left by deepfake generation processes
- Uses a compact yet powerful architecture with 4 convolutional layers
- Employs batch normalization and careful dropout for robust predictions
- Achieves high accuracy while maintaining real-time performance

### How It Detects Deepfakes:
1. Analyzes the mesoscopic properties of facial regions
2. Looks for inconsistencies in image generation patterns
3. Examines compression artifacts and visual coherence
4. Provides confidence scores based on accumulated evidence

## 💻 Technology Stack

### 🛠️ Core Components

- **🤖 MesoNet Model**
  - A specialized convolutional neural network
  - Optimized for real-time deepfake detection
  - Trained on extensive datasets of real and synthetic media

- **⚡ FastAPI**
  - Modern, fast web framework
  - Automatic API documentation
  - Type-safe request/response handling

- **🔌 Socket.IO**
  - Real-time WebSocket communication
  - Progress updates during analysis
  - Reliable bi-directional data flow

- **📊 TensorFlow**
  - Industry-standard deep learning framework
  - Optimized model inference
  - GPU acceleration support

- **👁️ OpenCV & MediaPipe**
  - Advanced face detection and tracking
  - Real-time video processing
  - Multi-face support with high accuracy

### 📚 Key Dependencies
- Python 3.11.9
- TensorFlow for model inference
- OpenCV (Headless) for video processing
- MediaPipe for primary face detection
- FastAPI + Uvicorn for the web server
- Socket.IO for real-time communication

## How It Works

### Deepfake Detection Process

1. **Face Detection**: 
   - Primary: MediaPipe Face Detection
   - Fallback: Haar Cascade Classifier
   - Adds padding around detected faces for better context

2. **Video Processing**:
   - Intelligent frame sampling across video thirds
   - Batch processing for performance
   - Supports both streaming chunks and direct uploads

3. **MesoNet Analysis**:
   - Preprocesses faces to 256x256 dimensions
   - Normalizes pixel values
   - Averages predictions across multiple frames
   - Provides confidence scores and individual frame predictions

### 🔌 API Endpoints

#### 🌐 WebSocket (`/socket.io/`)
- Real-time video processing with instant feedback
- Live progress updates during analysis
- Efficient chunked video upload support
- Reliable connection status monitoring
- Perfect for live video streams!

#### 🌍 HTTP REST
- `POST /upload`: Upload and analyze videos
  - Supports various video formats
  - Detailed analysis results
  - Performance metrics included

- `GET /health`: Service health check
  - Model status
  - System resources
  - Dependencies health

- `GET /`: API information
  - Version info
  - Available endpoints
  - Usage statistics

## ✨ Features

- **⚡ Real-time Processing**
  - Stream video chunks for instant analysis
  - Low-latency response times
  - Optimized for live video feeds

- **👥 Robust Face Detection**
  - Dual-method approach for reliability
  - Works with multiple face angles
  - Handles varying lighting conditions

- **📊 Detailed Analytics**
  - Frame-by-frame analysis
  - Confidence scores for each detection
  - Comprehensive result breakdown

- **🚀 Production Ready**
  - Enterprise-grade error handling
  - Extensive logging capabilities
  - Performance optimizations

- **📈 Progress Updates**
  - Real-time processing feedback
  - Step-by-step status updates
  - Time remaining estimates

- **🔧 Scalable Architecture**
  - Async processing for better performance
  - Efficient batch operations
  - Horizontally scalable design

## ⚡ Performance

- 🔄 Processes multiple frames concurrently
- 🎯 Intelligent frame sampling for longer videos
- 📦 Batch processing for improved throughput
- 📊 Progress tracking and detailed statistics
- ⚙️ Non-blocking async operations throughout

## 🔒 Security Features

- ✅ Comprehensive input validation
- ⚠️ Robust error handling
- 🔐 Secure file processing
- 🌐 CORS support for web integration
- 💪 Continuous health monitoring

## 🚀 Getting Started

Ready to detect deepfakes? Our API is designed to be developer-friendly and easy to integrate into your existing applications!

### Example Response

```json
{
  "prediction": "deepfake",
  "confidence": 0.98,
  "faces_detected": 1,
  "processing_time": "1.2s",
  "frame_analysis": {
    "total_frames": 30,
    "suspicious_frames": 28
  }
}
```

## 🌟 Why Choose Our Solution?

1. **Accuracy**: MesoNet's proven track record in deepfake detection
2. **Speed**: Real-time processing with optimized performance
3. **Scalability**: Handle multiple requests with ease
4. **Reliability**: Robust error handling and failsafes
5. **Integration**: Simple API with comprehensive documentation
