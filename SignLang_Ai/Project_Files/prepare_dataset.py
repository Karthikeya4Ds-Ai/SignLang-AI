import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# --- 1. Define Paths and Labels (MUST MATCH YOUR COLLECTION SCRIPT) ---

# This is the path to your single, unified data folder
DATA_PATH = 'C:/Users/parva/Desktop/SignLang_Ai/SignLang_Data_Dynamic'

# Define Static classes
STATIC_NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
STATIC_ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
# Define Dynamic classes
DYNAMIC_CLASSES = [
    'Hello', 'Goodbye', 'Thanks', 'Please', 'Yes', 'No',
    'Food', 'Water', 'Help', 'Stop', '_idle_'
]

# Combine all classes into the final 'actions' array
# This MUST be in the same order as your collection script
actions = np.array(STATIC_NUMBERS + STATIC_ALPHABET + DYNAMIC_CLASSES)
TOTAL_CLASSES = len(actions) # Should be 47

# Create the label map (e.g., '0' -> 0, 'A' -> 10, 'Hello' -> 36)
label_map = {label: num for num, label in enumerate(actions)}

print(f"Total classes to load: {TOTAL_CLASSES}")

# --- 2. Load All .npy Files ---
sequences, labels = [], []

print("Loading data from folders...")
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.isdir(action_path):
        print(f"Warning: Directory not found for {action}. Skipping.")
        continue
    
    # Get all .npy files
    sequence_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    
    print(f"Loading {len(sequence_files)} sequences for {action}...")
    
    for sequence_file in sequence_files:
        res = np.load(os.path.join(action_path, sequence_file))
        
        # Ensure it's the correct shape (60, 126)
        if res.shape == (60, 126):
            sequences.append(res)
            labels.append(label_map[action])
        else:
            print(f"  Warning: Skipping {sequence_file}, incorrect shape: {res.shape}. Expected (60, 126).")

print(f"\nData processing complete. Total sequences loaded: {len(sequences)}")

# --- 3. Finalize, Split, and Save ---
X = np.array(sequences)
y = to_categorical(np.array(labels), num_classes=TOTAL_CLASSES).astype(int)

# Split into 80% training and 20% testing
# stratify=y ensures that both training and testing sets have a proportional representation of all classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Data split complete.")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Save the four new master files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("---")
print("SUCCESS! Your master data files have been created:")
print("X_train.npy, y_train.npy, X_test.npy, y_test.npy")