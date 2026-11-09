import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- 1. Load Your Split Data ---
print("Loading split data files...")
try:
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
except FileNotFoundError:
    print("Error: Could not find X_train.npy, y_train.npy, X_test.npy, or y_test.npy")
    print("Please run 'prepare_dataset.py' first to create these files.")
    exit()

# Get configuration from loaded data
NUM_CLASSES = y_train.shape[1] # Should be 47
INPUT_SHAPE = (X_train.shape[1], X_train.shape[2]) # (60, 126)

print(f"Data loaded. Training model on {len(X_train)} samples.")
print(f"Input shape: {INPUT_SHAPE}, Total classes: {NUM_CLASSES}")

# --- 2. Define Model Architecture ---
print("Building the LSTM model...")

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=INPUT_SHAPE))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Final layer must match the total number of classes
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()

# --- 3. Define Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
log_dir = os.path.join('Logs')
tensorboard_callback = TensorBoard(log_dir=log_dir)

# --- 4. Compile and Train Model ---
print("Compiling model...")
# Use the stable learning rate
custom_optimizer = Adam(learning_rate=0.0001) 
model.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("--- Starting Model Training ---")
model.fit(X_train, y_train,
          epochs=100,
          batch_size=32,
          validation_data=(X_test, y_test),
          callbacks=[early_stopping, tensorboard_callback])

print("--- Training Complete ---")

# --- 5. Save Your Trained Model ---
model.save('sign_classifier_model.keras') # Save with a clear name
print("Model saved as 'sign_classifier_model.keras'")