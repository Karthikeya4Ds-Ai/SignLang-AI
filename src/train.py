"""
Model Training Pipeline
Supports stacked LSTM, GRU, and BiLSTM architectures with EarlyStopping,
learning rate reduction on plateau, TensorBoard logging, and automatic TFLite export.
"""

import os
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint

from src.config_loader import load_config, get_all_classes, get_project_root
from src.model import create_model, export_to_tflite
from src.dataset_builder import prepare_and_split_dataset
from src.evaluate import run_evaluation


def train(
    model_type: str = "gru",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    apply_normalization: bool = True
):
    """
    Executes complete training and validation cycle.
    """
    root = get_project_root()
    config = load_config()
    classes = get_all_classes(config)
    num_classes = len(classes)

    # 1. Load or Generate Dataset
    processed_npz = root / "data" / "processed" / "dataset_splits.npz"
    if processed_npz.is_file():
        print(f"Loading cached split data from: {processed_npz}")
        data = np.load(processed_npz)
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
    else:
        print("Processed splits not found. Running dataset builder...")
        X_train, X_test, y_train, y_test = prepare_and_split_dataset(
            config=config,
            apply_normalization=apply_normalization
        )

    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"\n--- Training Configuration ---")
    print(f"Model Type:    {model_type.upper()}")
    print(f"Input Shape:   {input_shape}")
    print(f"Num Classes:   {num_classes}")
    print(f"Train Samples: {len(X_train)}")
    print(f"Test Samples:  {len(X_test)}")
    print(f"Epochs:        {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")

    # 2. Build & Compile Model
    model = create_model(
        model_type=model_type,
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    model.summary()

    # 3. Callbacks
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / f"sign_classifier_{model_type}.keras"
    log_dir = root / "reports" / "logs" / model_type

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.get("model", {}).get("training", {}).get("early_stopping_patience", 20),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_categorical_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(log_dir=str(log_dir))
    ]

    # 4. Train
    print("\n--- Starting Training Loop ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Save final reference model
    model.save(best_model_path)
    print(f"\nBest model saved to: {best_model_path}")

    # 5. Export to TFLite
    try:
        tflite_path = models_dir / f"sign_classifier_{model_type}.tflite"
        export_to_tflite(model, str(tflite_path))
    except Exception as e:
        print(f"TFLite export notice: {e}")

    # 6. Run full evaluation
    print("\n--- Running Automated Evaluation ---")
    run_evaluation(model_path=str(best_model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignLang AI Classifier")
    parser.add_argument("--model_type", type=str, default="gru", choices=["gru", "lstm", "bilstm"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--no_norm", action="store_true", help="Disable wrist normalization")
    args = parser.parse_args()

    train(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        apply_normalization=not args.no_norm
    )
