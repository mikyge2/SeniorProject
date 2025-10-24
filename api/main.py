from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import uvicorn

from .api.routes import router, set_inference_engine
from .core.engine import ASLInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ASL Recognition API",
    description="Real-time American Sign Language recognition with Amharic translation.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the ASL inference engine on startup."""
    model_path = "export/asl_model.tflite"
    metadata_path = "processed_asl/metadata.json"

    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not Path(metadata_path).exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        inference_engine = ASLInferenceEngine(
            model_path=model_path,
            metadata_path=metadata_path,
            enable_amharic=True
        )
        set_inference_engine(inference_engine)
        logger.info("ASL Recognition API server started successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
