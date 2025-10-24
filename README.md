# Real-time American Sign Language (ASL) Recognition

This project is a real-time American Sign Language (ASL) recognition system featuring a machine learning model for inference and a FastAPI backend for serving the model.

## Project Structure

The repository is organized into the following directories:

- **`api/`**: Contains the FastAPI backend application.
  - **`api/`**: API endpoint definitions.
  - **`core/`**: Core components like the inference engine.
  - **`models/`**: Pydantic models for data validation.
  - **`services/`**: Services like the connection manager.
  - **`main.py`**: The main entry point for the FastAPI application.
- **`scripts/`**: Standalone Python scripts for tasks like model training, evaluation, and data preprocessing.
- **`export/`**: Contains the exported TensorFlow Lite model (`asl_model.tflite`).
- **`models/`**: Stores trained model artifacts.
- **`processed_asl/`**: Contains processed ASL data, including `metadata.json`.

## Getting Started

### Prerequisites

- Python 3.9+
- Pip

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/asl-recognition.git
    cd asl-recognition
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the FastAPI server, use the following command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be accessible at `http://localhost:8000`.

## Usage

### WebSocket API

The primary way to interact with the ASL recognition system is through the WebSocket API. The endpoint is:

```
ws://localhost:8000/ws/predict
```

You can send frames from a camera to this endpoint, and the server will return real-time predictions.

### REST API

A REST API is also available for single-frame predictions:

- **POST** `/predict`: Upload an image file to receive a prediction.

### Testing

A simple HTML client is available at `test-client.html` for testing the WebSocket connection and visualizing the predictions.

## Scripts

The `scripts/` directory contains several useful scripts:

- **`ModelTrainer.py`**: Trains the ASL recognition model.
- **`evaluateConfusion.py`**: Evaluates the model's performance.
- **`exportModel.py`**: Exports the trained model to TensorFlow Lite format.
- **`realtime_inference.py`**: A standalone script for real-time inference without the FastAPI server.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
