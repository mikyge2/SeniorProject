import os
import tensorflow as tf

# Load the trained model in Keras format
model = tf.keras.models.load_model("./models/asl/best_model.keras")

# Ensure export directory exists
os.makedirs("export", exist_ok=True)

# 1. Export as TensorFlow SavedModel (required for TFLite)
saved_model_path = "export/saved_model"
model.export(saved_model_path)
print(f"SavedModel exported to {saved_model_path}")

# 2. Convert the SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

tflite_path = "export/asl_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model exported to {tflite_path}")
