#!/usr/bin/env python3
"""
Production-Grade Sign Language Translator Trainer

A memory-efficient, modular trainer for sign language classification using
MobileNetV2 + hand landmarks architecture. Supports both alphabet (static images)
and word (sequence) datasets through TFRecord format.

Date: September 2025
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy

# Configure TensorFlow for optimal performance
tf.config.experimental.enable_op_determinism()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


class SignLanguageTrainer:
    """Manages the end-to-end training and evaluation of a sign language classification model.

    This class encapsulates the entire training pipeline, including data loading,
    model construction, training execution, evaluation, and results reporting.
    It uses a MobileNetV2 backbone combined with hand landmark data for a
    multi-modal approach to sign language recognition.

    Attributes:
        batch_size (int): The size of batches for training and evaluation.
        learning_rate (float): The initial learning rate for the Adam optimizer.
        num_classes (int): The number of distinct classes for classification.
        image_size (Tuple[int, int]): The target dimensions (height, width) for input images.
        num_landmarks (int): The number of hand landmark features (e.g., 42 for 21 2D points).
        history (Optional[tf.keras.callbacks.History]): Stores the training history.
        training_time (float): The total time taken for training, in seconds.
        model (Optional[Model]): The compiled Keras model.
    """

    def __init__(self,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 num_classes: int = 29,
                 image_size: Tuple[int, int] = (224, 224),
                 num_landmarks: int = 42):
        """Initializes the SignLanguageTrainer with training configurations.

        Args:
            batch_size: The batch size to use for training and evaluation.
            learning_rate: The learning rate for the Adam optimizer.
            num_classes: The total number of classes to predict.
            image_size: A tuple representing the (height, width) of input images.
            num_landmarks: The number of hand landmark coordinates (e.g., 21 points * 2 coords = 42).
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_landmarks = num_landmarks

        # Training metrics tracking
        self.history = None
        self.training_time = 0.0
        self.model = None

    def parse_tfrecord_fn(self, example_proto: tf.Tensor) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        """Parses a single TFRecord example into features and a one-hot encoded label.

        This function defines the structure of the TFRecord and decodes a
        serialized example into its constituent parts: an image, hand landmarks,
        and a label. The image is normalized to [0, 1], and the label is
        converted to a one-hot vector.

        Args:
            example_proto: A scalar string tensor representing a serialized
                `tf.train.Example` proto.

        Returns:
            A tuple containing:
            - A dictionary of features with 'image' and 'landmarks' tensors.
            - A one-hot encoded label tensor.
        """
        # Define the feature description
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/landmarks': tf.io.FixedLenFeature([self.num_landmarks], tf.float32),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'label_text': tf.io.FixedLenFeature([], tf.string),
        }

        # Parse the input tf.train.Example proto
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        # Decode and preprocess image
        image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        image = tf.ensure_shape(image, [*self.image_size, 3])

        # Normalize landmarks to [0, 1] (assuming they're already in reasonable ranges)
        landmarks = parsed_features['image/landmarks']
        landmarks = tf.clip_by_value(landmarks, 0.0, 1.0)  # Ensure [0, 1] range

        # One-hot encode labels
        label = tf.cast(parsed_features['label'], tf.int32)
        label_one_hot = tf.one_hot(label, depth=self.num_classes)

        # Return features and labels
        features = {
            'image': image,
            'landmarks': landmarks
        }

        return features, label_one_hot

    def load_dataset(self,
                     tfrecord_path: str,
                     is_training: bool = True,
                     shuffle_buffer: int = 1000,
                     seed: int = 47) -> tf.data.Dataset:
        """Loads and prepares a dataset from a TFRecord file.

        This method creates a `tf.data.Dataset` from a TFRecord file, parses
        the records using `parse_tfrecord_fn`, and sets up the dataset for
        training or evaluation. For training, it shuffles and repeats the
        dataset indefinitely. For all modes, it batches the data and prefetches
        for optimal performance.

        Args:
            tfrecord_path: The path to the `.tfrecord` file.
            is_training: If True, the dataset is shuffled and repeated.
            shuffle_buffer: The size of the buffer used for shuffling.
            seed: A random seed for shuffling to ensure reproducibility.

        Returns:
            A `tf.data.Dataset` object ready for use in `model.fit` or
            `model.evaluate`.
        """
        # Load TFRecord dataset
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        # Parse records
        dataset = dataset.map(
            self.parse_tfrecord_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if is_training:
            # Shuffle and repeat for training with fixed seed for reproducibility
            dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
            dataset = dataset.repeat()

        # Batch the dataset
        dataset = dataset.batch(self.batch_size)

        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def build_model(self) -> Model:
        """Builds and compiles the sign language classification model.

        The model architecture consists of two main input branches:
        1. An image branch using a pre-trained MobileNetV2 model (frozen) for
           feature extraction from hand images.
        2. A landmark branch that processes normalized hand landmark coordinates.

        These two feature sets are concatenated and passed through a dense
        classification head with dropout for regularization.

        Returns:
            A compiled Keras `Model` instance ready for training.
        """
        # Input layers
        image_input = layers.Input(shape=(*self.image_size, 3), name='image')
        landmarks_input = layers.Input(shape=(self.num_landmarks,), name='landmarks')

        # MobileNetV2 backbone (frozen for feature extraction)
        mobilenet_base = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'  # Use global average pooling
        )
        mobilenet_base.trainable = False  # Freeze backbone

        # Extract image features
        image_features = mobilenet_base(image_input)

        # Combine image features and landmarks
        combined_features = layers.Concatenate(name='feature_fusion')([
            image_features, landmarks_input
        ])

        # Classification head
        x = layers.Dense(128, activation='relu', name='dense_1')(combined_features)
        x = layers.Dropout(0.5, name='dropout_1')(x)

        # Output layer
        predictions = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)

        # Create model
        model = Model(
            inputs=[image_input, landmarks_input],
            outputs=predictions,
            name='sign_language_classifier'
        )

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=CategoricalCrossentropy(),
            metrics=[
                CategoricalAccuracy(name='accuracy'),
                TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
            ]
        )

        return model

    def setup_callbacks(self, save_dir: str) -> list:
        """Configures and returns a list of Keras callbacks for training.

        This method sets up a standard suite of callbacks to enhance the training
        process, including:
        - `ModelCheckpoint`: To save the best model based on validation accuracy.
        - `EarlyStopping`: To halt training if validation accuracy plateaus.
        - `TensorBoard`: To log metrics for visualization.
        - `ReduceLROnPlateau`: To decrease the learning rate if validation loss stagnates.

        Args:
            save_dir: The directory where model checkpoints and logs will be saved.

        Returns:
            A list of configured `tf.keras.callbacks.Callback` instances.
        """
        # Ensure save directory exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Model checkpoint callback
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )

        # Early stopping callback
        early_stopping_callback = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )

        # TensorBoard logging
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'tensorboard_logs'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )

        # Learning rate reduction
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )

        return [
            checkpoint_callback,
            early_stopping_callback,
            tensorboard_callback,
            lr_scheduler
        ]

    def calculate_steps(self, tfrecord_path: str) -> int:
        """Calculates the number of steps per epoch for a given dataset.

        This is determined by counting the total number of records in the
        TFRecord file and dividing by the batch size.

        Args:
            tfrecord_path: The path to the `.tfrecord` file.

        Returns:
            The integer number of steps (batches) per epoch.
        """
        # Count records in TFRecord file
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        num_samples = sum(1 for _ in dataset)

        return max(1, num_samples // self.batch_size)

    def train(self,
              train_tfrecord: str,
              val_tfrecord: str,
              epochs: int,
              save_dir: str) -> Dict[str, Any]:
        """Orchestrates the model training process.

        This method performs the following steps:
        1. Loads the training and validation datasets.
        2. Calculates the number of steps per epoch for each dataset.
        3. Builds and compiles the model.
        4. Sets up the training callbacks.
        5. Calls `model.fit` to run the training loop.
        6. Saves the training history to a JSON file.

        Args:
            train_tfrecord: The path to the training `.tfrecord` file.
            val_tfrecord: The path to the validation `.tfrecord` file.
            epochs: The total number of epochs to train for.
            save_dir: The directory to save the model and training logs.

        Returns:
            A dictionary containing the training history, total training time,
            and final accuracy metrics.
        """
        print("🚀 Starting Sign Language Model Training...")
        print(f"📊 Configuration:")
        print(f"   - Batch Size: {self.batch_size}")
        print(f"   - Learning Rate: {self.learning_rate}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Classes: {self.num_classes}")

        # Load datasets
        print("\n📂 Loading datasets...")
        train_dataset = self.load_dataset(train_tfrecord, is_training=True)
        val_dataset = self.load_dataset(val_tfrecord, is_training=False)

        # Calculate steps
        train_steps = self.calculate_steps(train_tfrecord)
        val_steps = self.calculate_steps(val_tfrecord)

        print(f"   - Training steps per epoch: {train_steps}")
        print(f"   - Validation steps per epoch: {val_steps}")

        # Build model
        print("\n🏗️ Building model architecture...")
        self.model = self.build_model()

        # Print model summary
        print("\n📋 Model Summary:")
        self.model.summary()

        # Setup callbacks
        training_callbacks = self.setup_callbacks(save_dir)

        # Start training
        print(f"\n🔥 Training started...")
        start_time = time.time()

        self.history = self.model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=val_dataset,
            validation_steps=val_steps,
            callbacks=training_callbacks,
            verbose=1
        )

        self.training_time = time.time() - start_time

        # Save training history
        history_path = os.path.join(save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {k: [float(x) for x in v] for k, v in self.history.history.items()}
            json.dump(history_dict, f, indent=2)

        print(f"\n✅ Training completed in {self.training_time:.2f} seconds")
        print(f"📈 Training history saved to: {history_path}")

        return {
            'history': self.history.history,
            'training_time': self.training_time,
            'final_train_accuracy': self.history.history['accuracy'][-1],
            'final_val_accuracy': self.history.history['val_accuracy'][-1]
        }

    def evaluate(self, test_tfrecord: str) -> Dict[str, float]:
        """Evaluates the trained model on the test dataset.

        Args:
            test_tfrecord: The path to the test `.tfrecord` file.

        Returns:
            A dictionary containing the evaluation metrics (e.g., loss,
            accuracy, top_3_accuracy).

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        print("\n🧪 Evaluating model on test dataset...")

        # Load test dataset
        test_dataset = self.load_dataset(test_tfrecord, is_training=False, shuffle_buffer=1000, seed=42)
        test_steps = self.calculate_steps(test_tfrecord)

        # Evaluate model
        test_results = self.model.evaluate(
            test_dataset,
            steps=test_steps,
            verbose=1,
            return_dict=True
        )

        print(f"📊 Test Results:")
        for metric, value in test_results.items():
            print(f"   - {metric}: {value:.4f}")

        return test_results

    def print_training_summary(self, results: Dict[str, Any], test_results: Optional[Dict[str, float]] = None):
        """Prints a formatted summary of the training and evaluation results.

        Args:
            results: The dictionary returned by the `train` method.
            test_results: An optional dictionary of test results from the
                `evaluate` method.
        """
        print("\n" + "=" * 60)
        print("🎯 TRAINING SUMMARY")
        print("=" * 60)

        print(f"⏱️  Total Training Time: {results['training_time']:.2f} seconds")
        print(f"📈 Final Training Accuracy: {results['final_train_accuracy']:.4f}")
        print(f"📊 Final Validation Accuracy: {results['final_val_accuracy']:.4f}")

        if test_results:
            print(f"🧪 Test Accuracy: {test_results.get('accuracy', 0):.4f}")
            print(f"🎯 Test Top-3 Accuracy: {test_results.get('top_3_accuracy', 0):.4f}")

        # Performance indicators
        val_acc = results['final_val_accuracy']
        if val_acc > 0.95:
            print("🌟 Excellent performance!")
        elif val_acc > 0.90:
            print("✅ Good performance!")
        elif val_acc > 0.80:
            print("⚠️  Moderate performance - consider more training")
        else:
            print("❌ Poor performance - check data and hyperparameters")

        print("=" * 60)


def main():
    """Main function to run the training pipeline from the command line.

    Parses command-line arguments, initializes the `SignLanguageTrainer`,
    runs the training and evaluation processes, and prints a final summary.
    """

    parser = argparse.ArgumentParser(
        description="Production-Grade Sign Language Translator Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--train-tfrecord', required=True,
                        help='Path to training TFRecord file')
    parser.add_argument('--val-tfrecord', required=True,
                        help='Path to validation TFRecord file')
    parser.add_argument('--save-dir', required=True,
                        help='Directory to save model checkpoints and logs')

    # Optional arguments
    parser.add_argument('--test-tfrecord',
                        help='Path to test TFRecord file (optional)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--num-classes', type=int, default=29,
                        help='Number of classification classes')

    args = parser.parse_args()

    # Validate input files
    for tfrecord_path in [args.train_tfrecord, args.val_tfrecord]:
        if not os.path.exists(tfrecord_path):
            raise FileNotFoundError(f"TFRecord file not found: {tfrecord_path}")

    if args.test_tfrecord and not os.path.exists(args.test_tfrecord):
        raise FileNotFoundError(f"Test TFRecord file not found: {args.test_tfrecord}")

    # Initialize trainer
    trainer = SignLanguageTrainer(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_classes=args.num_classes
    )

    # Train model
    training_results = trainer.train(
        train_tfrecord=args.train_tfrecord,
        val_tfrecord=args.val_tfrecord,
        epochs=args.epochs,
        save_dir=args.save_dir
    )

    # Evaluate on test set if provided
    test_results = None
    if args.test_tfrecord:
        test_results = trainer.evaluate(args.test_tfrecord)

    # Print final summary
    trainer.print_training_summary(training_results, test_results)

    print(f"\n🎉 Training pipeline completed successfully!")
    print(f"📁 Model and logs saved to: {args.save_dir}")


if __name__ == "__main__":
    main()