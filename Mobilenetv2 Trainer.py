import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
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
    LEARNING_RATE = 0.0005  # Reduced learning rate
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    PREFETCH_BUFFER = 2
    SHUFFLE_BUFFER = 1000
    QUANTIZE_MODEL = True
    TARGET_SIZE_MB = 10
    USE_MIXED_PRECISION = False  # Disable mixed precision for debugging
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

# Disable mixed precision for now to avoid issues
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
# FIXED DATA LOADER
# ==============================
class FixedMemoryEfficientDataLoader:
    def __init__(self, config: Config):
        self.config = config

        # Check if files exist before proceeding
        if not os.path.exists(self.config.LABELS_FILE):
            raise FileNotFoundError(f"Labels file not found: {self.config.LABELS_FILE}")

        self.label_map = self.load_labels()
        self.num_classes = len(self.label_map)
        self.batch_files = sorted(glob.glob(self.config.BATCH_FILES_PATTERN))

        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found matching pattern: {self.config.BATCH_FILES_PATTERN}")

        logger.info(f"Found {len(self.batch_files)} batch files")
        logger.info(f"Number of classes: {self.num_classes}")

        # Load all data info upfront for proper splitting
        self.all_samples = self._load_all_sample_info()
        self.dataset_info = self._analyze_dataset()

    def load_labels(self) -> Dict[int, str]:
        """Load label mapping from JSON file"""
        try:
            with open(self.config.LABELS_FILE, 'r') as f:
                label_map = json.load(f)
            return {int(k): v for k, v in label_map.items()}
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            raise

    def _analyze_dataset(self) -> Dict:
        """Analyze dataset composition"""
        total_samples = len(self.all_samples)
        class_counts = np.zeros(self.num_classes, dtype=int)

        for _, _, label in self.all_samples:
            if label < self.num_classes:
                class_counts[label] += 1

        test_size = int(total_samples * self.config.TEST_SPLIT)
        val_size = int(total_samples * self.config.VALIDATION_SPLIT)
        train_size = total_samples - test_size - val_size

        logger.info(f"Dataset analysis:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Train samples: {train_size}")
        logger.info(f"  Validation samples: {val_size}")
        logger.info(f"  Test samples: {test_size}")

        return {
            'total_samples': total_samples,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'class_counts': class_counts,
        }

    def _load_all_sample_info(self):
        """Load info about all samples for proper stratified splitting"""
        all_samples = []
        logger.info("Loading sample information from batch files...")

        for i, batch_file in enumerate(self.batch_files):
            try:
                data = np.load(batch_file)
                y_batch = data['y']
                for j, label in enumerate(y_batch):
                    all_samples.append((batch_file, j, int(label)))
                data.close()
                logger.info(f"  Loaded {len(y_batch)} samples from {batch_file}")
            except Exception as e:
                logger.error(f"Error loading {batch_file}: {e}")
                raise

        logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples

    def validate_data_splits(self, y_train, y_val, y_test):
        """Validate that all classes are represented in each split"""
        train_classes = set(y_train)
        val_classes = set(y_val)
        test_classes = set(y_test)

        logger.info(f"Classes in train: {len(train_classes)}/{self.num_classes}")
        logger.info(f"Classes in val: {len(val_classes)}/{self.num_classes}")
        logger.info(f"Classes in test: {len(test_classes)}/{self.num_classes}")

        if len(train_classes) < self.num_classes * 0.8:  # At least 80% of classes
            logger.warning("Some classes missing from training set!")

        return True

    def _stratified_split(self):
        """Create stratified train/val/test splits"""
        try:
            # Separate samples by class for stratification
            X_info = [(file_path, idx) for file_path, idx, label in self.all_samples]
            y_labels = [label for _, _, label in self.all_samples]

            logger.info("Creating stratified splits...")

            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_info, y_labels,
                test_size=self.config.TEST_SPLIT,
                stratify=y_labels,
                random_state=42
            )

            # Second split: separate train and validation
            val_size_adjusted = self.config.VALIDATION_SPLIT / (1 - self.config.TEST_SPLIT)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                stratify=y_temp,
                random_state=42
            )

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error in stratified split: {e}")
            raise

    def _create_tf_dataset(self, sample_list, labels, is_training=False):
        """Create TensorFlow dataset from sample list"""

        def data_generator():
            # Group samples by file for efficient loading
            file_groups = {}
            for (file_path, idx), label in zip(sample_list, labels):
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append((idx, label))

            # Load and yield data
            epoch_count = 0
            while True:  # For training, repeat indefinitely
                file_items = list(file_groups.items())
                if is_training:
                    np.random.shuffle(file_items)

                samples_yielded = 0
                for file_path, samples in file_items:
                    try:
                        data = np.load(file_path)
                        X_batch = data['X']

                        sample_indices = [idx for idx, _ in samples]
                        if is_training:
                            np.random.shuffle(sample_indices)

                        for idx in sample_indices:
                            sample_idx, label = next((s for s in samples if s[0] == idx), (None, None))
                            if sample_idx is not None:
                                X_sample = X_batch[sample_idx].astype(np.float32)
                                y_one_hot = tf.keras.utils.to_categorical(label, self.num_classes)
                                yield X_sample, y_one_hot
                                samples_yielded += 1

                        data.close()
                    except Exception as e:
                        logger.error(f"Error loading data from {file_path}: {e}")
                        continue

                epoch_count += 1
                if not is_training:  # For validation/test, run once per epoch
                    logger.debug(f"Completed epoch {epoch_count} for {'val/test'} with {samples_yielded} samples")
                    break

        # Create dataset
        try:
            dataset = tf.data.Dataset.from_generator(
                data_generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.num_classes,), dtype=tf.float32)
                )
            )

            if is_training:
                dataset = dataset.batch(self.config.BATCH_SIZE, drop_remainder=True)
            else:
                dataset = dataset.batch(self.config.BATCH_SIZE)

            return dataset.prefetch(tf.data.AUTOTUNE)
        except Exception as e:
            logger.error(f"Error creating TensorFlow dataset: {e}")
            raise

    def create_datasets(self):
        """Create train, validation, and test datasets"""
        try:
            logger.info("Creating stratified data splits...")

            X_train, X_val, X_test, y_train, y_val, y_test = self._stratified_split()

            logger.info(f"Train samples: {len(X_train)}")
            logger.info(f"Validation samples: {len(X_val)}")
            logger.info(f"Test samples: {len(X_test)}")

            # Validate splits
            self.validate_data_splits(y_train, y_val, y_test)

            # Create datasets
            logger.info("Creating TensorFlow datasets...")
            train_ds = self._create_tf_dataset(X_train, y_train, is_training=True)
            val_ds = self._create_tf_dataset(X_val, y_val, is_training=False)
            test_ds = self._create_tf_dataset(X_test, y_test, is_training=False)

            # Calculate class weights
            class_counts = np.bincount(y_train, minlength=self.num_classes)
            total_samples = len(y_train)
            class_weights = {}
            for i, count in enumerate(class_counts):
                if count > 0:
                    class_weights[i] = total_samples / (self.num_classes * count)
                else:
                    class_weights[i] = 1.0

            logger.info("Datasets created successfully!")

            return train_ds, val_ds, test_ds, {
                'class_weights': class_weights,
                'label_map': self.label_map,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }

        except Exception as e:
            logger.error(f"Error creating datasets: {e}")
            raise


# ==============================
# IMPROVED MODEL WITH BETTER ARCHITECTURE
# ==============================
class ImprovedMobileSignLanguageModel:
    def __init__(self, config: Config, num_classes: int):
        self.config = config
        self.num_classes = num_classes

    def create_model(self):
        try:
            logger.info("Creating MobileNetV2 model...")

            # Use MobileNetV2 (more stable than V3 for this task)
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3),
                alpha=1.0,
                include_top=False,
                weights='imagenet'
            )

            # Freeze base model initially
            base_model.trainable = False

            inputs = tf.keras.Input(shape=(self.config.IMG_SIZE, self.config.IMG_SIZE, 3))

            # Preprocessing
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

            # Base model
            x = base_model(x, training=False)

            # Custom head with better regularization
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.1)(x)

            # Output layer
            outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)

            model = tf.keras.Model(inputs, outputs, name=self.config.MODEL_NAME)

            logger.info("Model created successfully!")
            return model

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise

    def compile_model(self, model):
        try:
            logger.info("Compiling model...")
            # Use a smaller learning rate initially
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy']
            )
            logger.info("Model compiled successfully!")
            return model
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise


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
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=self.config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join(self.config.LOGS_DIR, f'{model_name}_{timestamp}')
            )
        ]

    def train_model(self, model, train_ds, val_ds, class_weights, steps_per_epoch, validation_steps):
        try:
            logger.info("Starting model training...")
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
            logger.info("Training completed!")
            return model
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise


# ==============================
# EVALUATION
# ==============================
class ModelEvaluator:
    def __init__(self, config: Config, label_map: Dict[int, str]):
        self.config = config
        self.label_map = label_map

    def evaluate_model(self, model, test_ds, test_steps):
        try:
            logger.info("Evaluating model...")
            results = model.evaluate(test_ds, steps=test_steps, verbose=1)
            logger.info(f"Evaluation results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise


# ==============================
# MOBILE EXPORT
# ==============================
class MobileExporter:
    def __init__(self, config: Config):
        self.config = config

    def export_for_mobile(self, model, model_name):
        try:
            logger.info("Exporting model for mobile...")
            tflite_path = os.path.join(self.config.MOBILE_EXPORT_DIR, f'{model_name}.tflite')
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"Model exported to {tflite_path}")
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise


# ==============================
# MAIN
# ==============================
def main():
    try:
        logger.info("Starting ASL training pipeline...")
        logger.info(f"TensorFlow version: {tf.__version__}")

        # Use fixed data loader
        data_loader = FixedMemoryEfficientDataLoader(config)
        train_ds, val_ds, test_ds, data_info = data_loader.create_datasets()

        config.NUM_CLASSES = len(data_info['label_map'])

        # Calculate proper steps
        steps_per_epoch = data_info['train_size'] // config.BATCH_SIZE
        validation_steps = data_info['val_size'] // config.BATCH_SIZE
        test_steps = data_info['test_size'] // config.BATCH_SIZE

        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        logger.info(f"Test steps: {test_steps}")

        # Use improved model
        model_builder = ImprovedMobileSignLanguageModel(config, config.NUM_CLASSES)
        model = model_builder.create_model()
        model = model_builder.compile_model(model)

        # Print model summary
        model.summary()

        # Train with fixed setup
        trainer = TrainingManager(config)
        model = trainer.train_model(
            model, train_ds, val_ds,
            data_info['class_weights'],
            steps_per_epoch, validation_steps
        )

        # Evaluate
        evaluator = ModelEvaluator(config, data_info['label_map'])
        evaluator.evaluate_model(model, test_ds, test_steps)

        # Export
        exporter = MobileExporter(config)
        exporter.export_for_mobile(model, config.MODEL_NAME)

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()