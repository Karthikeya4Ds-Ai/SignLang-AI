import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Setup ---

# 1. Path for exported data
DATA_PATH = os.path.join("C:\\Users\\parva\\Desktop\\SignLang_Ai\\SignLang_Data_Dynamic") 

# 2. Define the list of dynamic classes so we can give instructions
DYNAMIC_CLASSES = [
    'Hello', 'Goodbye', 'Thanks', 'Please', 'Yes', 'No',
    'Food', 'Water', 'Help', 'Stop', '_idle_'  # The "nothing" class
]

# 3. Define Static classes
STATIC_NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
STATIC_ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# 4. Combine all classes into the final 'actions' array
actions = np.array(STATIC_NUMBERS + STATIC_ALPHABET + DYNAMIC_CLASSES)
# ===================================================================


# 5. Number of sequences (videos) for each action
no_sequences = 50

# 6. Number of frames per sequence (60 is good)
sequence_length = 60

# --- MediaPipe Setup ---
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

# ===================================================================
# == Action Selection & Resume Menu ==
# ===================================================================
print("--- Sign Language Data Collection ---")
print(f"Found {len(actions)} total actions to record.")
print("Loading actions and checking progress...")

first_incomplete_index = 0
found_first_incomplete = False

# Loop through actions to create folders and display status
for i, action in enumerate(actions):
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)
    
    num_sequences_done = 0
    try:
        existing_files = os.listdir(action_path)
        npy_files = [f for f in existing_files if f.endswith('.npy') and f.split('.')[0].isdigit()]
        num_sequences_done = len(npy_files)
    except Exception as e:
        print(f"Could not read directory {action_path}: {e}")

    status = "COMPLETE" if num_sequences_done == no_sequences else f"{num_sequences_done}/{no_sequences}"
    print(f"[{i}] {action} ({status})")
    
    if num_sequences_done < no_sequences and not found_first_incomplete:
        first_incomplete_index = i
        found_first_incomplete = True

print(f"\nPress Enter to start from the first incomplete action ('{actions[first_incomplete_index]}')")
choice = input(f"Or enter a number (0-{len(actions)-1}) to start from: ")

start_index = first_incomplete_index
if choice.strip().isdigit():
    try:
        choice_int = int(choice.strip())
        if 0 <= choice_int < len(actions):
            start_index = choice_int
        else:
            print(f"Invalid number. Starting from {actions[first_incomplete_index]}.")
    except ValueError:
        print(f"Invalid input. Starting from {actions[first_incomplete_index]}.")

print(f"\nStarting with action: {actions[start_index]}")
time.sleep(2)

# Slice the actions array to start from the user's choice
actions_to_record = actions[start_index:]
# ===================================================================


# --- Main Collection Loop ---

cap = cv2.VideoCapture(0)  # Change the index if necessary to select the correct camera
quit_flag = False

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for action in actions_to_record:
        action_path = os.path.join(DATA_PATH, action)
        
        # Resume from the next available sequence number
        start_sequence = 0
        try:
            existing_files = os.listdir(action_path)
            existing_nums = [int(f.split('.')[0]) for f in existing_files if f.endswith('.npy') and f.split('.')[0].isdigit()]
            if existing_nums:
                start_sequence = max(existing_nums) + 1
        except Exception as e:
            print(f"Error checking existing sequences for {action}: {e}")
        
        if start_sequence >= no_sequences:
            print(f"Skipping {action}, already have {no_sequences} sequences.")
            continue

        for sequence in range(start_sequence, no_sequences):
            
            # --- 1. Wait for user ready ---
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                text = f"READY: '{action.upper()}' Video {sequence+1}/{no_sequences}"
                cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "PRESS [SPACE] TO START, [Q] TO QUIT", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Instruction for static vs dynamic
                if action in DYNAMIC_CLASSES:
                    cv2.putText(frame, "DO THE MOTION", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "HOLD THE POSE STILL", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed', frame)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    quit_flag = True
                    break
            
            if quit_flag:
                break

            # --- Countdown before recording ---
            for i in range(2, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                text = f"STARTING IN {i}..."
                cv2.rectangle(frame, (0,0), (640, 40), (0, 0, 255), -1)
                cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1000) # Wait 1 second

            # --- 2. Start recording frames ---
            sequence_data = []
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                cv2.rectangle(frame, (0,0), (640, 40), (0, 255, 0), -1)
                cv2.putText(image, f'RECORDING... FRAME {frame_num+1}/{sequence_length}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                sequence_data.append(keypoints)
                cv2.waitKey(10)

            # --- 3. Save the sequence ---
            sequence_data = np.array(sequence_data)
            save_path = os.path.join(DATA_PATH, action, f"{sequence}.npy")
            np.save(save_path, sequence_data)
            print(f"Saved: {save_path}")
        
        if quit_flag:
            break

print("Quitting application and releasing camera.")
cap.release()
cv2.destroyAllWindows()