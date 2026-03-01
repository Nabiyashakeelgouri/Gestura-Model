# live_capture_prototype.py
import cv2
import time
import numpy as np
import mediapipe as mp
import json
from collections import deque

# Mediapipe utilities
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Parameters
SAVE_LANDMARKS = True   # set False to disable saving sequences
SEQ_LEN = 30            # frames per recorded sequence
OUTPUT_SEQ_DIR = "recorded_sequences"  # will be created

# prepare webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Close other apps using camera or try another index.")

# For FPS
prev = 0

# For recording sequences
recording = False
seq_buffer = deque(maxlen=SEQ_LEN)
seq_count = 0

# init mediapipe
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6) as hands, \
     mp_face.FaceMesh(static_image_mode=False,
                      max_num_faces=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.6) as face_mesh:
        # force the window to appear in front and set size
    cv2.namedWindow("Live - Mediapipe", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live - Mediapipe", 900, 600)

    print("Starting camera...")


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # detect hands and face
        hand_res = hands.process(frame_rgb)
        face_res = face_mesh.process(frame_rgb)

        frame_rgb.flags.writeable = True
        out_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # draw hands
        if hand_res.multi_hand_landmarks:
            for hand_landmarks in hand_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    out_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style())

        # draw face mesh (if detected)
        if face_res.multi_face_landmarks:
            for face_landmarks in face_res.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    out_frame,
                    face_landmarks,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,128,255), thickness=1)
)
        # compute FPS
        curr = time.time()
        fps = 1 / (curr - prev) if (curr - prev) > 0 else 0
        prev = curr
        cv2.putText(out_frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # show instructions
        cv2.putText(out_frame, "Press 'r' to start/stop record seq, 's' to save snapshot, 'q' to quit",
                    (10, out_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # if recording, collect landmarks (both hands & face coords)
        if SAVE_LANDMARKS and recording:
            frame_landmarks = {"hands": [], "face": []}
            if hand_res.multi_hand_landmarks:
                for h in hand_res.multi_hand_landmarks:
                    coords = [(lm.x, lm.y, lm.z) for lm in h.landmark]
                    frame_landmarks["hands"].append(coords)
            if face_res.multi_face_landmarks:
                for f in face_res.multi_face_landmarks:
                    coords = [(lm.x, lm.y, lm.z) for lm in f.landmark]
                    frame_landmarks["face"].append(coords)
            seq_buffer.append(frame_landmarks)
            cv2.putText(out_frame, f"Recording: {len(seq_buffer)}/{SEQ_LEN}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if len(seq_buffer) == SEQ_LEN:
                # save sequence to disk
                import os
                os.makedirs(OUTPUT_SEQ_DIR, exist_ok=True)
                seq_path = os.path.join(OUTPUT_SEQ_DIR, f"seq_{seq_count}.json")
                with open (seq_path, "w") as f:
                    json.dump(list(seq_buffer), f)
                print(f"Saved sequence to {seq_path}")
                seq_count += 1
                seq_buffer.clear()
                recording = False

        cv2.imshow("Live - Mediapipe", out_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # save snapshot
            import os
            os.makedirs("snapshots", exist_ok=True)
            ts = int(time.time())
            cv2.imwrite(f"snapshots/snap_{ts}.png", out_frame)
            print("Snapshot saved.")
        elif key == ord('r'):
            recording = not recording
            if recording:
                seq_buffer.clear()
                print("Started recording sequence...")
            else:
                print("Stopped recording.")

cap.release()
cv2.destroyAllWindows()
