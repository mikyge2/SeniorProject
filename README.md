# Real-Time American Sign Language (ASL) Recognition System

This project is a comprehensive, real-time American Sign Language (ASL) recognition system designed to translate sign language gestures into text and speech. It includes a full pipeline for data preprocessing, model training, and deployment via a low-latency FastAPI backend, making it suitable for mobile applications.

## Key Features

- **End-to-End ASL Pipeline**: From data preparation to a deployable API.
- **High-Performance Model**: Utilizes a fine-tuned machine learning model for accurate gesture recognition.
- **Real-Time Inference**: Optimized for low-latency processing of live camera feeds.
- **FastAPI Backend**: A robust and scalable backend server to handle real-time requests via WebSockets.
- **Amharic Translation**: Built-in support for translating recognized English words into Amharic.
- **Text-to-Speech (TTS)**: Vocalizes the recognized words and their translations.
- **Modular and Extensible**: The codebase is organized into clear, reusable components.

## Project Structure

```
.
├── export/                     # Exported models (e.g., .tflite)
├── models/                     # Saved trained models
├── processed_asl/              # Processed datasets and metadata
├── FASTAPIBackend.py           # FastAPI server for the ASL recognition API
├── ModelTrainer.py             # Script for training the ASL recognition model
├── preprocessor.py             # Script for preprocessing the ASL dataset
├── enhanced_asl_inference.py   # Core real-time inference logic
├── realtime_inference.py       # Standalone script for real-time inference (without API)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── ...                         # Other utility and testing scripts
```

## Getting Started

### 1. Prerequisites

- Python 3.8+
- OpenCV
- TensorFlow / TensorFlow Lite
- MediaPipe

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### 3. Data Preprocessing

Before training the model, you need to preprocess your ASL dataset. The `preprocessor.py` script handles this by augmenting the data, extracting features, and preparing it for training.

**Usage:**

```bash
python preprocessor.py --data_path /path/to/your/asl_dataset --output_path ./processed_asl
```

- `--data_path`: Path to the raw ASL dataset (e.g., `asl_alphabet_train`).
- `--output_path`: Directory to save the processed data and metadata.

### 4. Model Training

Once the data is preprocessed, you can train the model using `ModelTrainer.py`. This script trains a model on the processed data and saves the best-performing version.

**Usage:**

```bash
python ModelTrainer.py --data_path ./processed_asl --model_output_path ./models
```

- `--data_path`: Path to the processed data from the previous step.
- `--model_output_path`: Directory to save the trained model files.

### 5. Exporting the Model for Inference

After training, the model needs to be converted to a TensorFlow Lite (`.tflite`) format for efficient inference, especially on mobile devices.

**Usage:**

```bash
python exportModel.py --model_path ./models/asl_model.h5 --output_path ./export
```

- `--model_path`: Path to the trained Keras model (`.h5` file).
- `--output_path`: Directory to save the exported `.tflite` model.

## Usage

This system can be used in two primary ways: as a standalone real-time inference application or via the FastAPI backend for integration with other services (like mobile apps).

### 1. Standalone Real-Time Inference

The `realtime_inference.py` script provides a way to run the ASL recognition model directly on a live camera feed.

**Usage:**

```bash
python realtime_inference.py --model_path ./export/asl_model.tflite --metadata_path ./processed_asl/metadata.json
```

This will open a window showing your camera feed with the recognized ASL letter and the currently formed word.

### 2. FastAPI Backend

For use in production or with a mobile application, the `FASTAPIBackend.py` script launches a web server with API endpoints for inference. It supports both WebSocket for real-time communication and standard HTTP requests.

**To start the server:**

```bash
uvicorn FASTAPIBackend:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

#### API Endpoints

- **`GET /health`**: Health check for the server.
- **`WS /ws/predict`**: WebSocket endpoint for real-time frame processing.
- **`POST /predict`**: REST endpoint for single-frame predictions.
- **`POST /reset/{connection_id}`**: Resets the word tracker for a given connection.

For detailed information on API usage, request/response formats, and mobile integration examples, please see the `FASTAPIBackendREADME.md` file.

## Utility Scripts

The repository also includes several utility scripts for development and evaluation:

- `checkOverlap.py`: Checks for overlap between training and testing datasets.
- `evaluateConfusion.py`: Evaluates the model and generates a confusion matrix.
- `inspectPreprocessed.py`: A tool to visualize and inspect the preprocessed data.
- `predictImage.py`: Predicts the ASL sign from a single static image file.
- `testSpeech.py`: A simple script to test the Text-to-Speech functionality.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.