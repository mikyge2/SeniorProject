# FastAPI ASL Recognition Server

Real-time American Sign Language recognition API with Amharic translation support for mobile applications.

## Features

- **Real-time ASL recognition** via WebSocket or REST API
- **Letter-to-word tracking** with intelligent word completion
- **Amharic translation** for completed words
- **Word suggestions** from online dictionary
- **Mobile-optimized** with low-latency processing
- **Scalable WebSocket connections** for multiple simultaneous users

## Quick Start

### 1. Installation

```bash
# Clone your project and install dependencies
pip install fastapi uvicorn websockets python-multipart
pip install opencv-python mediapipe tensorflow
pip install requests deep-translator gtts pygame pillow

# Install the original inference script dependencies
pip install numpy pathlib
```

### 2. Project Structure

```
your-project/
├── main.py                    # FastAPI server (this file)
├── enhanced_asl_inference.py  # Original inference engine
├── export/
│   └── asl_model.tflite      # Your trained model
├── processed_asl/
│   └── metadata.json         # Class mapping metadata
└── README.md
```

### 3. Start the Server

```bash
# Development mode with auto-reload
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

The server will start at `http://localhost:8000`

### 4. Test the API

Visit `http://localhost:8000` for API documentation, or check health:

```bash
curl http://localhost:8000/health
```

## API Endpoints

### Health Check
```http
GET /health
```
Returns server status and connection count.

### WebSocket (Recommended)
```http
WS /ws/predict
```
Real-time frame processing with persistent word tracking.

### REST Fallback
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
connection_id: <optional_id>
```

### Reset Word Tracker
```http
POST /reset/{connection_id}
```

## Mobile App Integration

### WebSocket Connection Example

**JavaScript/React Native:**
```javascript
const ws = new WebSocket('ws://your-server:8000/ws/predict');

ws.onopen = () => {
    console.log('Connected to ASL API');
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    
    if (response.type === 'prediction') {
        const prediction = response.data;
        
        // Update UI with current letter and word
        updateUI({
            letter: prediction.letter,
            confidence: prediction.confidence,
            currentWord: prediction.current_word,
            suggestions: prediction.word_suggestions,
            letterProgress: prediction.letter_progress
        });
        
        // Handle completed words
        if (prediction.word_completed) {
            onWordCompleted(
                prediction.word_completed,
                prediction.amharic_translation
            );
        }
    }
};

// Send camera frame
function sendFrame(imageBase64) {
    ws.send(JSON.stringify({
        type: 'frame',
        data: imageBase64  // Base64 encoded image
    }));
}

// Reset current word
function resetWord() {
    ws.send(JSON.stringify({
        type: 'reset'
    }));
}
```

**Flutter/Dart:**
```dart
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class ASLService {
  late WebSocketChannel _channel;
  
  void connect() {
    _channel = WebSocketChannel.connect(
      Uri.parse('ws://your-server:8000/ws/predict')
    );
    
    _channel.stream.listen((message) {
      final response = json.decode(message);
      
      if (response['type'] == 'prediction') {
        final prediction = response['data'];
        handlePrediction(prediction);
      }
    });
  }
  
  void sendFrame(String base64Image) {
    _channel.sink.add(json.encode({
      'type': 'frame',
      'data': base64Image
    }));
  }
  
  void resetWord() {
    _channel.sink.add(json.encode({'type': 'reset'}));
  }
}
```

### Camera Frame Capture

**React Native (Expo Camera):**
```javascript
import { Camera } from 'expo-camera';

const CameraScreen = () => {
  const [camera, setCamera] = useState(null);
  
  // Capture and send frames at 5 FPS
  useEffect(() => {
    const interval = setInterval(async () => {
      if (camera) {
        const photo = await camera.takePictureAsync({
          base64: true,
          quality: 0.7,
          skipProcessing: true
        });
        
        // Send to API
        sendFrame(photo.base64);
      }
    }, 200); // 5 FPS
    
    return () => clearInterval(interval);
  }, [camera]);
  
  return (
    <Camera
      style={{ flex: 1 }}
      ref={setCamera}
      type={Camera.Constants.Type.front}
    />
  );
};
```

## Response Format

### Real-time Prediction Response
```json
{
  "type": "prediction",
  "data": {
    "letter": "H",
    "confidence": 0.94,
    "current_word": "HEL",
    "word_suggestions": ["HELLO", "HELP", "HELD"],
    "word_finalized": false,
    "word_completed": null,
    "amharic_translation": null,
    "letter_progress": 0.75,
    "timestamp": "2024-01-15T10:30:45.123456",
    "processing_time_ms": 45.2,
    "frame_count": 156
  }
}
```

### Word Completion Response
```json
{
  "type": "prediction", 
  "data": {
    "letter": "O",
    "confidence": 0.89,
    "current_word": "",
    "word_suggestions": [],
    "word_finalized": true,
    "word_completed": "HELLO",
    "amharic_translation": "ሰላም",
    "letter_progress": 0.0,
    "timestamp": "2024-01-15T10:30:47.654321",
    "processing_time_ms": 52.1,
    "frame_count": 167
  }
}
```

## Performance Considerations

### Server-side
- **Latency target:** <200ms per frame
- **Throughput:** Supports multiple concurrent WebSocket connections
- **GPU acceleration:** Automatically used if TensorFlow detects compatible GPU
- **Memory management:** Each connection maintains its own WordTracker state

### Mobile-side
- **Frame rate:** Recommend 5-10 FPS to balance responsiveness and bandwidth
- **Image quality:** Use 70% JPEG quality for optimal size/clarity trade-off
- **Connection management:** Implement reconnection logic for network interruptions

### Network Optimization
```javascript
// Example: Adaptive frame rate based on connection quality
let frameRate = 5; // Start at 5 FPS

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  const processingTime = response.data?.processing_time_ms;
  
  // Adjust frame rate based on server performance
  if (processingTime > 200) {
    frameRate = Math.max(2, frameRate - 1);
  } else if (processingTime < 100) {
    frameRate = Math.min(10, frameRate + 1);
  }
  
  // Update capture interval
  updateCaptureInterval(1000 / frameRate);
};
```

## Configuration

### Environment Variables
```bash
# Model paths
MODEL_PATH=export/asl_model.tflite
METADATA_PATH=processed_asl/metadata.json

# Features
ENABLE_AMHARIC=true
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
```

### Word Tracker Settings
Modify in `main.py`:
```python
word_tracker = WordTracker(
    window_size=6,           # Frames to consider for stability
    confidence_threshold=0.6, # Minimum confidence to register
    pause_threshold=2.0,     # Seconds of pause to finalize word
    min_letter_duration=0.8  # Seconds to hold letter
)
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use a reverse proxy (nginx) for WebSocket connections
- Enable SSL/TLS for secure connections
- Monitor memory usage with multiple concurrent connections
- Consider horizontal scaling with Redis for shared state

## Troubleshooting

### Common Issues

**Model not found:**
```
FileNotFoundError: Model file not found: export/asl_model.tflite
```
- Ensure your model file is in the correct path
- Update `MODEL_PATH` in startup configuration

**MediaPipe initialization failed:**
- Install required dependencies: `pip install mediapipe opencv-python`
- Check camera permissions on the mobile device

**WebSocket connection refused:**
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall/network settings
- Ensure correct WebSocket URL format: `ws://` not `http://`

**High latency (>200ms):**
- Reduce image quality/size on mobile side
- Check server CPU/GPU utilization
- Consider reducing frame rate to 3-5 FPS

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## License

This project uses the same license as your original ASL inference system.