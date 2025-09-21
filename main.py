"""
Production-Ready Deepfake Detection API
FastAPI + Socket.IO with MediaPipe face detection
"""

import asyncio
import numpy as np
import tensorflow as tf
import pickle
import os
import tempfile
import time
import base64
import logging
import json
import uuid
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Set environment variables for headless operation (must be before cv2 import)
os.environ['OPENCV_OPENCL_RUNTIME'] = ''
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['DISPLAY'] = ':99' if 'DISPLAY' not in os.environ else os.environ['DISPLAY']

# Now import cv2 after setting environment variables
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import socketio

from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/Meso4_DF.h5")
LABEL_PATH = os.getenv("LABEL_PATH", "models/mesonet_idx_to_class.pkl")
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
IMG_DIMENSIONS = {'height': 256, 'width': 256, 'channels': 3}

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global detector instance
detector = None

# Create Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    async_mode='asgi',
    logger=True,
    engineio_logger=True
)

# Store video chunks per client
video_chunks = {}

class Classifier:
    """Base classifier class"""
    def __init__(self):
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    """MesoNet-4 model for deepfake detection"""
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                          loss='mean_squared_error',
                          metrics=['accuracy'])

    def init_model(self):
        """Initialize MesoNet-4 architecture"""
        x = Input(shape=(IMG_DIMENSIONS['height'],
                        IMG_DIMENSIONS['width'],
                        IMG_DIMENSIONS['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

class DeepfakeDetector:
    """Production-ready deepfake detection class"""
    
    def __init__(self, model_path: str, label_path: str):
        # Load MesoNet model
        self.model = Meso4()
        self.model.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        
        # Load label mapping
        with open(label_path, 'rb') as f:
            self.idx_to_class = pickle.load(f)
        logger.info(f"Labels loaded: {self.idx_to_class}")
        
        # Initialize face detection
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize MediaPipe
        self.mediapipe_detector = None
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mediapipe_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe face detector loaded")
        except ImportError:
            logger.warning("MediaPipe not available, using Haar Cascade only")
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("DeepfakeDetector initialized successfully")

    def detect_faces_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces using MediaPipe"""
        try:
            if self.mediapipe_detector is None:
                return None
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_frame.shape
            
            results = self.mediapipe_detector.process(rgb_frame)
            
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Add padding
                padding = 0.1
                x_pad = int(width * padding)
                y_pad = int(height * padding)
                x = max(0, x - x_pad)
                y = max(0, y - y_pad)
                width = min(width + 2 * x_pad, w - x)
                height = min(height + 2 * y_pad, h - y)
                
                face = rgb_frame[y:y+height, x:x+width]
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                return face_bgr if face_bgr.size > 0 else None
        except Exception as e:
            logger.error(f"MediaPipe error: {e}")
        return None

    def detect_faces_haar(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect faces using Haar Cascade (fallback)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                
                # Add padding
                padding = 0.1
                x_pad = int(w * padding)
                y_pad = int(h * padding)
                x = max(0, x - x_pad)
                y = max(0, y - y_pad)
                w = min(w + 2 * x_pad, frame.shape[1] - x)
                h = min(h + 2 * y_pad, frame.shape[0] - y)
                
                face = frame[y:y+h, x:x+w]
                return face if face.size > 0 else None
        except Exception as e:
            logger.error(f"Haar cascade error: {e}")
        return None

    def extract_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract face using MediaPipe first, then Haar Cascade"""
        face = self.detect_faces_mediapipe(frame)
        if face is None:
            face = self.detect_faces_haar(frame)
        return face

    def sample_frames(self, total_frames: int, target_frames: int = 40) -> List[int]:
        """Sample frames across video thirds"""
        if total_frames <= target_frames:
            return list(range(total_frames))
        
        indices = []
        third = total_frames // 3
        
        # Sample from each third
        frames_per_third = target_frames // 3
        for third_idx in range(3):
            start = third_idx * third
            end = start + third if third_idx < 2 else total_frames
            step = max(1, (end - start) // frames_per_third)
            
            for i in range(frames_per_third):
                idx = start + i * step
                if idx < end:
                    indices.append(idx)
        
        return sorted(list(set(indices)))[:target_frames]

    async def process_video(self, video_bytes: bytes, progress_callback=None) -> Tuple[List[np.ndarray], dict]:
        """Process video and extract faces with progress updates"""
        start_time = time.time()
        
        # Save video bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name
        
        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if progress_callback:
                await progress_callback({
                    "stage": "video_loaded",
                    "total_frames": total_frames,
                    "duration": duration,
                    "fps": fps
                })
            
            frame_indices = self.sample_frames(total_frames)
            faces = []
            
            if progress_callback:
                await progress_callback({
                    "stage": "processing_frames",
                    "frames_to_process": len(frame_indices)
                })
            
            # Process frames in batches
            batch_size = 8
            for i in range(0, len(frame_indices), batch_size):
                batch_indices = frame_indices[i:i+batch_size]
                batch_faces = await self._process_frame_batch(cap, batch_indices)
                faces.extend(batch_faces)
                
                if progress_callback:
                    await progress_callback({
                        "stage": "batch_complete",
                        "processed": min(i + batch_size, len(frame_indices)),
                        "total": len(frame_indices),
                        "faces_found": len(faces)
                    })
            
            cap.release()
            
            processing_stats = {
                'frames_processed': len(frame_indices),
                'faces_detected': len(faces),
                'processing_time': time.time() - start_time,
                'video_duration': duration,
                'total_frames': total_frames,
                'fps': fps
            }
            
            return faces, processing_stats
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    async def _process_frame_batch(self, cap, frame_indices):
        """Process batch of frames for face detection"""
        batch_faces = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                loop = asyncio.get_event_loop()
                face = await loop.run_in_executor(self.executor, self.extract_face, frame)
                if face is not None:
                    batch_faces.append(face)
        return batch_faces

    def preprocess_faces(self, faces: List[np.ndarray]) -> np.ndarray:
        """Preprocess faces for model input"""
        if not faces:
            return np.array([])
        
        processed_faces = []
        for face in faces:
            face_resized = cv2.resize(face, (256, 256))
            face_normalized = face_resized.astype(np.float32) / 255.0
            processed_faces.append(face_normalized)
        
        return np.array(processed_faces)

    async def predict(self, faces: List[np.ndarray]) -> Tuple[str, float, List[float]]:
        """Predict deepfake using MesoNet model"""
        if not faces:
            return "Error", 0.0, []
        
        # Preprocess faces
        processed_faces = self.preprocess_faces(faces)
        
        # Run prediction in executor to avoid blocking
        loop = asyncio.get_event_loop()
        predictions = await loop.run_in_executor(
            self.executor, self.model.predict, processed_faces
        )
        
        prediction_scores = predictions.flatten().tolist()
        avg_score = np.mean(prediction_scores)
        
        # Threshold at 0.5
        final_label = self.idx_to_class[int(avg_score > 0.5)]
        confidence = avg_score if final_label == 'Real' else 1 - avg_score
        
        return final_label, confidence, prediction_scores

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize detector on startup"""
    global detector
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        if not os.path.exists(LABEL_PATH):
            logger.error(f"Label file not found: {LABEL_PATH}")
            raise FileNotFoundError(f"Label file not found: {LABEL_PATH}")
        
        detector = DeepfakeDetector(MODEL_PATH, LABEL_PATH)
        logger.info("Detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        detector = None
    
    yield
    
    logger.info("Shutting down...")
    if detector and detector.executor:
        detector.executor.shutdown(wait=True)

# Create FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="Production-ready deepfake detection with WebSocket support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    logger.info(f"Client {sid} connected")
    await sio.emit('connection_established', {
        'sid': sid,
        'status': 'connected',
        'detector_ready': detector is not None
    }, to=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    logger.info(f"Client {sid} disconnected")
    # Clean up any incomplete video chunks for this client
    if sid in video_chunks:
        del video_chunks[sid]

@sio.event
async def video_chunk(sid, data):
    """Handle video chunks sent by client"""
    try:
        chunk_id = data.get('chunk_id', 0)
        total_chunks = data.get('total_chunks')
        chunk_data = data.get('chunk')
        is_last_chunk = data.get('is_last', False)
        
        if not chunk_data:
            await sio.emit('chunk_error', {
                'error': 'No chunk data provided',
                'chunk_id': chunk_id,
                'timestamp': time.time()
            }, to=sid)
            return
        
        # Initialize storage for this client if needed
        if sid not in video_chunks:
            video_chunks[sid] = {}
        
        # Store chunk
        video_chunks[sid][chunk_id] = chunk_data
        
        # Send chunk acknowledgment
        await sio.emit('chunk_received', {
            'chunk_id': chunk_id,
            'chunks_received': len(video_chunks[sid]),
            'total_expected': total_chunks,
            'timestamp': time.time()
        }, to=sid)
        
        # Check if all chunks received
        if (total_chunks and len(video_chunks[sid]) == total_chunks) or is_last_chunk:
            # Reconstruct complete video
            try:
                # Sort chunks by chunk_id and concatenate
                sorted_chunks = [video_chunks[sid][i] for i in sorted(video_chunks[sid].keys())]
                complete_video_base64 = ''.join(sorted_chunks)
                
                # Process complete video
                await process_complete_video(sid, complete_video_base64)
                
                # Clean up chunks
                del video_chunks[sid]
                
            except Exception as e:
                logger.error(f"Error reconstructing video for client {sid}: {e}")
                await sio.emit('reconstruction_error', {
                    'error': f'Failed to reconstruct video: {str(e)}',
                    'timestamp': time.time()
                }, to=sid)
                # Clean up on error
                if sid in video_chunks:
                    del video_chunks[sid]
    
    except Exception as e:
        logger.error(f"Error handling video chunk for client {sid}: {e}")
        await sio.emit('chunk_error', {
            'error': str(e),
            'timestamp': time.time()
        }, to=sid)

async def process_complete_video(sid, complete_video_base64):
    """Process complete video reconstructed from chunks"""
    if detector is None:
        await sio.emit('prediction_error', {
            'error': 'Detector not initialized',
            'timestamp': time.time()
        }, to=sid)
        return
    
    try:
        # Progress callback function
        async def progress_callback(progress_data):
            await sio.emit('prediction_progress', {
                **progress_data,
                'timestamp': time.time()
            }, to=sid)
        
        # Remove data URL prefix if present
        if ',' in complete_video_base64:
            complete_video_base64 = complete_video_base64.split(',')[1]
        
        # Decode video
        video_bytes = base64.b64decode(complete_video_base64)
        logger.info(f"Processing reconstructed video for {sid}: {len(video_bytes)} bytes")
        
        await sio.emit('prediction_started', {
            'message': 'Starting video processing (from chunks)',
            'video_size': len(video_bytes),
            'timestamp': time.time()
        }, to=sid)
        
        # Process video with progress updates
        faces, processing_stats = await detector.process_video(
            video_bytes, progress_callback
        )
        
        if not faces:
            await sio.emit('prediction_complete', {
                'error': 'No faces detected in video',
                'processing_stats': processing_stats,
                'timestamp': time.time()
            }, to=sid)
            return
        
        await progress_callback({
            'stage': 'running_prediction',
            'faces_count': len(faces)
        })
        
        # Run prediction
        final_label, confidence, individual_scores = await detector.predict(faces)
        
        # Prepare result
        result = {
            'prediction': final_label,
            'confidence': float(confidence),
            'num_faces_processed': len(faces),
            'individual_scores': individual_scores,
            'avg_score': float(np.mean(individual_scores)),
            'processing_stats': {
                **processing_stats,
                'faces_per_second': len(faces) / processing_stats['processing_time'] if processing_stats['processing_time'] > 0 else 0
            },
            'timestamp': time.time(),
            'source': 'chunked_upload'
        }
        
        logger.info(f"Chunk-based prediction complete for {sid}: {final_label} ({confidence:.4f})")
        
        # Send final result
        await sio.emit('prediction_complete', result, to=sid)
        
    except Exception as e:
        logger.error(f"Error processing reconstructed video for {sid}: {e}")
        await sio.emit('prediction_error', {
            'error': str(e),
            'timestamp': time.time()
        }, to=sid)

@sio.event
async def predict_video(sid, data):
    """Handle video prediction requests via WebSocket"""
    logger.info(f"Received video prediction request from {sid}")
    
    if detector is None:
        await sio.emit('prediction_error', {
            'error': 'Detector not initialized',
            'timestamp': time.time()
        }, to=sid)
        return
    
    if 'video' not in data:
        await sio.emit('prediction_error', {
            'error': 'No video data provided',
            'timestamp': time.time()
        }, to=sid)
        return
    
    try:
        # Progress callback function
        async def progress_callback(progress_data):
            await sio.emit('prediction_progress', {
                **progress_data,
                'timestamp': time.time()
            }, to=sid)
        
        # Decode video data
        video_base64 = data['video']
        if ',' in video_base64:
            video_base64 = video_base64.split(',')[1]
        
        video_bytes = base64.b64decode(video_base64)
        logger.info(f"Processing video: {len(video_bytes)} bytes")
        
        await sio.emit('prediction_started', {
            'message': 'Starting video processing',
            'video_size': len(video_bytes),
            'timestamp': time.time()
        }, to=sid)
        
        # Process video with progress updates
        faces, processing_stats = await detector.process_video(
            video_bytes, progress_callback
        )
        
        if not faces:
            await sio.emit('prediction_complete', {
                'error': 'No faces detected in video',
                'processing_stats': processing_stats,
                'timestamp': time.time()
            }, to=sid)
            return
        
        await progress_callback({
            'stage': 'running_prediction',
            'faces_count': len(faces)
        })
        
        # Run prediction
        final_label, confidence, individual_scores = await detector.predict(faces)
        
        # Prepare result
        result = {
            'prediction': final_label,
            'confidence': float(confidence),
            'num_faces_processed': len(faces),
            'individual_scores': individual_scores,
            'avg_score': float(np.mean(individual_scores)),
            'processing_stats': {
                **processing_stats,
                'faces_per_second': len(faces) / processing_stats['processing_time'] if processing_stats['processing_time'] > 0 else 0
            },
            'timestamp': time.time(),
            'source': 'single_upload'
        }
        
        logger.info(f"Prediction complete: {final_label} ({confidence:.4f})")
        
        # Send final result
        await sio.emit('prediction_complete', result, to=sid)
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        await sio.emit('prediction_error', {
            'error': str(e),
            'timestamp': time.time()
        }, to=sid)

@sio.event
async def ping(sid, data):
    """Handle ping requests"""
    await sio.emit('pong', {
        'timestamp': time.time(),
        'detector_ready': detector is not None
    }, to=sid)

# HTTP endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Deepfake Detection API",
        "version": "1.0.0",
        "status": "running",
        "detector_ready": detector is not None,
        "endpoints": {
            "websocket": "Connect to /socket.io/ for real-time detection",
            "http_upload": "POST /upload for file upload",
            "health": "GET /health for health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if detector is not None else "unhealthy",
        "detector_loaded": detector is not None,
        "model_path": MODEL_PATH,
        "label_path": LABEL_PATH,
        "timestamp": time.time()
    }

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """HTTP endpoint for video upload"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        video_bytes = await file.read()
        logger.info(f"Processing uploaded file: {file.filename} ({len(video_bytes)} bytes)")
        
        # Process video
        faces, processing_stats = await detector.process_video(video_bytes)
        
        if not faces:
            return JSONResponse(content={
                "error": "No faces detected in video",
                "processing_stats": processing_stats,
                "timestamp": time.time()
            })
        
        # Run prediction
        final_label, confidence, individual_scores = await detector.predict(faces)
        
        result = {
            "filename": file.filename,
            "prediction": final_label,
            "confidence": float(confidence),
            "num_faces_processed": len(faces),
            "individual_scores": individual_scores,
            "avg_score": float(np.mean(individual_scores)),
            "processing_stats": {
                **processing_stats,
                "faces_per_second": len(faces) / processing_stats['processing_time'] if processing_stats['processing_time'] > 0 else 0
            },
            "timestamp": time.time()
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    import uvicorn
    
    # Railway uses PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting Deepfake Detection Server on port {port}...")
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
