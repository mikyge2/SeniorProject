import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
from datetime import datetime
import glob
from typing import Dict, List, Tuple, Generator

# ==============================
# CONFIG
# ==============================
class Config:
    BATCH_FILES_PATTERN = "asl_dataset_batch_*.npz"
    LABELS_FILE = "asl_labels.json"
    IMG_SIZE = 224
    NUM_CLASSES = 29
    MODEL_NAME = "MobileSignLanguage"
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    PREFETCH_BUFFER = 2
    SHUFFLE_BUFFER = 1000
    QUANTIZE_MODEL = True
    TARGET_SIZE_MB = 10
    USE_MIXED_PRECISION = True
    EARLY_STOPPING_PATIENCE = 8
    REDUCE_LR_PATIENCE = 4
    SAVE_BEST_ONLY = True
    MODEL_SAVE_DIR = "models"
    LOGS_DIR = "training_logs"
    PLOTS_DIR = "plots"
    MOBILE_EXPORT_DIR = "mobile_models"

config = Config()
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(config.LOGS_DIR, exist_ok=True)
os.makedirs(config.PLOTS_DIR, exist_ok=True)
os.makedirs(config.MOBILE_EXPORT_DIR, exist_ok=True)

if config.USE_MIXED_PRECISION:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{config.LOGS_DIR}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# MEMORY-EFFICIENT DATA LOADER
# ==============================
class MemoryEfficientDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.label_map = self.load_labels()
        self.num_classes = len(self.label_map)
        self.batch_files = sorted(glob.glob(self.config.BATCH_FILES_PATTERN))
        self.dataset_info = self._analyze_dataset()

    def load_labels(self) -> Dict[int, str]:
        with open(self.config.LABELS_FILE, 'r') as f:
            label_map = json.load(f)
        return {int(k): v for k, v in label_map.items()}

    def _analyze_dataset(self) -> Dict:
        total_samples = 0
        class_counts = np.zeros(self.num_classes, dtype=int)
        if not self.batch_files:
            raise FileNotFoundError("No batch files found")

        for batch_file in self.batch_files:
            data = np.load(batch_file)
            y_batch = data['y']
            total_samples += len(y_batch)
            unique, counts = np.unique(y_batch, return_counts=True)
            for class_idx, count in zip(unique, counts):
                if class_idx < self.num_classes:
                    class_counts[class_idx] += count
            data.close()

        test_size = int(total_samples * self.config.TEST_SPLIT)
        val_size = int(total_samples * self.config.VALIDATION_SPLIT)
        train_size = total_samples - test_size - val_size

        return {
            'total_samples': total_samples,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'class_counts': class_counts,
        }

    def _create_file_index(self) -> List[Tuple[str, int, int]]:
        file_index = []
        for batch_file in self.batch_files:
            data = np.load(batch_file)
            batch_size = len(data['y'])
            file_index.append((batch_file, 0, batch_size))
            data.close()
        return file_index

    def _split_file_index(self, file_index: List[Tuple[str, int, int]]):
        total_samples = sum(end - start for _, start, end in file_index)
        test_size = int(total_samples * self.config.TEST_SPLIT)
        val_size = int(total_samples * self.config.VALIDATION_SPLIT)
        all_indices = list(range(total_samples))
        np.random.shuffle(all_indices)
        test_indices = set(all_indices[:test_size])
        val_indices = set(all_indices[test_size:test_size + val_size])
        train_indices = set(all_indices[test_size + val_size:])

        train_files, val_files, test_files = [], [], []
        current_idx = 0
        for file_path, start, end in file_index:
            file_train_indices, file_val_indices, file_test_indices = [], [], []
            for local_idx in range(start, end):
                global_idx = current_idx + local_idx
                if global_idx in train_indices:
                    file_train_indices.append(local_idx)
                elif global_idx in val_indices:
                    file_val_indices.append(local_idx)
                elif global_idx in test_indices:
                    file_test_indices.append(local_idx)
            if file_train_indices:
                train_files.append((file_path, file_train_indices))
            if file_val_indices:
                val_files.append((file_path, file_val_indices))
            if file_test_indices:
                test_files.append((file_path, file_test_indices))
            current_idx += (end - start)
        return train_files, val_files, test_files

    def _data_generator(self, file_list: List[Tuple[str, List[int]]]) -> Generator:
        while True:
            np.random.shuffle(file_list)
            for file_path, indices in file_list:
                data = np.load(file_path)
                X_batch = data['X']
                y_batch = data['y']
                shuffled_indices = np.random.permutation(indices)
                for idx in shuffled_indices:
                    X_sample = X_batch[idx]
                    y_sample = y_batch[idx]
                    y_one_hot = tf.keras.utils.to_categorical(y_sample, self.num_classes)
                    yield X_sample, y_one_hot
                data.close()

    def create_datasets(self):
        file_index = self._create_file_index()
        train_files, val_files, test_files = self._split_file_index(file_index)
        train_ds = tf.data.Dataset.from_generator(
            lambda: self._data_generator(train_files),
            output_signature=(
                tf.TensorSpec(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
            )
        ).batch(self.config.BATCH_SIZE).prefetch(self.config.PREFETCH_BUFFER)
        val_ds = tf.data.Dataset.from_generator(
            lambda: self._data_generator(val_files),
            output_signature=(
                tf.TensorSpec(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
            )
        ).batch(self.config.BATCH_SIZE).prefetch(self.config.PREFETCH_BUFFER)
        test_ds = tf.data.Dataset.from_generator(
            lambda: self._data_generator(test_files),
            output_signature=(
                tf.TensorSpec(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
            )
        ).batch(self.config.BATCH_SIZE).prefetch(self.config.PREFETCH_BUFFER)

        class_counts = self.dataset_info['class_counts']
        total_train_val = self.dataset_info['total_samples'] - self.dataset_info['test_size']
        class_weights = {i: (total_train_val / (self.num_classes * count)) if count > 0 else 1.0 for i, count in enumerate(class_counts)}

        return train_ds, val_ds, test_ds, {
            'class_weights': class_weights,
            'label_map': self.label_map,
            'train_size': self.dataset_info['train_size'],
            'val_size': self.dataset_info['val_size'],
            'test_size': self.dataset_info['test_size']
        }

# ==============================
# MODEL
# ==============================
class MobileSignLanguageModel:
    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes

    def create_model(self):
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3),
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        inputs = tf.keras.Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3))
        x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        model = tf.keras.Model(inputs, outputs, name=self.config.MODEL_NAME)
        return model

    def compile_model(self, model):
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        return model

# ==============================
# TRAINING
# ==============================
class TrainingManager:
    def __init__(self, config: Config):
        self.config = config
        self.history = None

    def create_callbacks(self, model_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return [
            callbacks.ModelCheckpoint(
                filepath=os.path.join(self.config.MODEL_SAVE_DIR, f'{model_name}_best_{timestamp}.keras'),
                monitor='val_accuracy',
                save_best_only=self.config.SAVE_BEST_ONLY,
                verbose=1
            ),
            callbacks.EarlyStopping(monitor='val_accuracy', patience=self.config.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.config.REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1),
            callbacks.TensorBoard(log_dir=os.path.join(self.config.LOGS_DIR, f'{model_name}_{timestamp}'))
        ]

    def train_model(self, model, train_ds, val_ds, class_weights, steps_per_epoch, validation_steps):
        self.history = model.fit(
            train_ds,
            epochs=self.config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=self.create_callbacks(self.config.MODEL_NAME),
            verbose=1
        )
        return model

# ==============================
# EVALUATION
# ==============================
class ModelEvaluator:
    def __init__(self, config: Config, label_map: Dict[int, str]):
        self.config = config
        self.label_map = label_map

    def evaluate_model(self, model, test_ds, test_steps):
        results = model.evaluate(test_ds, steps=test_steps, verbose=1)
        return results

# ==============================
# MOBILE EXPORT
# ==============================
class MobileExporter:
    def __init__(self, config: Config):
        self.config = config

    def export_for_mobile(self, model, model_name):
        tflite_path = os.path.join(self.config.MOBILE_EXPORT_DIR, f'{model_name}.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"Model exported to {tflite_path}")

# ==============================
# MAIN
# ==============================
def main():
    try:
        data_loader = MemoryEfficientDataLoader(config)
        train_ds, val_ds, test_ds, data_info = data_loader.create_datasets()
        config.NUM_CLASSES = len(data_info['label_map'])
        steps_per_epoch = max(1, data_info['train_size'] // config.BATCH_SIZE)
        validation_steps = max(1, data_info['val_size'] // config.BATCH_SIZE)
        test_steps = max(1, data_info['test_size'] // config.BATCH_SIZE)

        model_builder = MobileSignLanguageModel(config, config.NUM_CLASSES)
        model = model_builder.create_model()
        model = model_builder.compile_model(model)

        trainer = TrainingManager(config)
        model = trainer.train_model(model, train_ds, val_ds, data_info['class_weights'], steps_per_epoch, validation_steps)

        evaluator = ModelEvaluator(config, data_info['label_map'])
        evaluator.evaluate_model(model, test_ds, test_steps)

        exporter = MobileExporter(config)
        exporter.export_for_mobile(model, config.MODEL_NAME)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()
