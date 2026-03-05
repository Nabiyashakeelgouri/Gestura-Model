from ai_engine.capture.camera import start_camera, read_frame, release_camera
from ai_engine.preprocessing.landmark_extractor import extract_landmarks
import cv2
import os
import json

cap = start_camera()
activated = False
sequence = []
SEQUENCE_LENGTH = 30

def save_sequence(sequence, gesture_name):
    base_path = "recorded_sequences"
    gesture_path = os.path.join(base_path, gesture_name)

    os.makedirs(gesture_path, exist_ok=True)

    existing_files = len(os.listdir(gesture_path))
    file_path = os.path.join(gesture_path, f"seq_{existing_files}.json")

    with open(file_path, "w") as f:
        json.dump(sequence, f)

    print(f"Saved: {file_path}")

while True:
    frame = read_frame(cap)
    if frame is None:
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        activated = True
        sequence = []

    if activated:
        landmarks = extract_landmarks(frame)
        sequence.append(landmarks)

        cv2.putText(frame, f"Collecting: {len(sequence)}/30",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        if len(sequence) == SEQUENCE_LENGTH:
            gesture_name = "stop"  # change manually while recording
            save_sequence(sequence, gesture_name)
            activated = False

    cv2.imshow("Video Test", frame)

    if key == ord('q'):
        break

release_camera(cap)