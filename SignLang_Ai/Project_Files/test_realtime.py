import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf

# --- 1. Load Your Trained Model ---
print("Loading model...")
model = tf.keras.models.load_model('sign_classifier_model.keras')
print("Model loaded successfully!")

# --- 2. Define Your Class Names (CRITICAL) ---
# This list MUST be in the exact same order as your training scripts

STATIC_NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
STATIC_ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]
DYNAMIC_CLASSES = [
    'Hello', 'Goodbye', 'Thanks', 'Please', 'Yes', 'No',
    'Food', 'Water', 'Help', 'Stop', '_idle_'
]
ALL_CLASSES = np.array(STATIC_NUMBERS + STATIC_ALPHABET + DYNAMIC_CLASSES)


# --- 3. Setup MediaPipe and Helper Functions ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# --- 4. Setup Real-Time Prediction Variables ---
SEQUENCE_LENGTH = 60 # 60 frames
CONFIDENCE_THRESHOLD = 0.7 # Only show predictions above 70% confidence
sequence = [] # This will hold the last 60 frames of data
current_action = '' # The word to display on screen

# --- 5. Main Real-Time Loop ---
cap = cv2.VideoCapture(0) # Your camera index
if not cap.isOpened():
    print("Error: Could not open webcam. Check camera index (e.g., 0, 1, or 2).")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # Flip horizontally

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        
        # --- Prediction Logic ---
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:] # Keep only the last 60 frames
        
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0) 
            res = model.predict(input_data, verbose=0)[0] 
            predicted_index = np.argmax(res)
            confidence = res[predicted_index]
            
            if confidence > CONFIDENCE_THRESHOLD:
                predicted_class_name = ALL_CLASSES[predicted_index]
                
                if predicted_class_name != current_action:
                    current_action = predicted_class_name
            
            # If the most likely class is '_idle_', always clear the action
            if ALL_CLASSES[predicted_index] == '_idle_':
                current_action = ''
            
        # --- Display the prediction on the screen ---
        cv2.rectangle(image, (0,0), (640, 40), (0, 0, 0), -1) 
        cv2.putText(image, current_action, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()