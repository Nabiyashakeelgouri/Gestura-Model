import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch

from ai_engine.preprocessing.landmark_extractor import extract_landmarks
from ai_engine.recognition.dynamic_dataset import flatten_dynamic_frame
from ai_engine.recognition.dynamic_model import DynamicGestureModel, load_dynamic_label_map


ROOT_DIR = Path(__file__).resolve().parents[2]
DYNAMIC_MODEL_PATH = ROOT_DIR / "models" / "dynamic_model.pth"
SEQUENCE_LENGTH = 30
MIN_HAND_FRAMES_IN_SEQUENCE = 12
PREDICTION_SMOOTHING_WINDOW = 10
PREDICTION_CONFIDENCE_THRESHOLD = 0.70


def _load_label_map() -> dict:
    label_map = load_dynamic_label_map()
    if label_map:
        return label_map
    return {"hello": 0, "no": 1, "stop": 2, "thank_u": 3, "yes": 4}


label_map = _load_label_map()
reverse_map = {v: k for k, v in label_map.items()}


def _load_model() -> DynamicGestureModel:
    state_dict = torch.load(DYNAMIC_MODEL_PATH, map_location=torch.device("cpu"))
    loaded_model = DynamicGestureModel(num_classes=len(label_map))
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()
    return loaded_model


model = _load_model()

sequence_buffer: List[List[float]] = []
hand_presence_buffer: List[int] = []
recent_predictions: List[str] = []
last_prediction_confidence: float = 0.0


def reset_dynamic_state() -> None:
    global sequence_buffer, hand_presence_buffer, recent_predictions, last_prediction_confidence
    sequence_buffer = []
    hand_presence_buffer = []
    recent_predictions = []
    last_prediction_confidence = 0.0


def _most_frequent_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def get_last_prediction_info() -> Tuple[Optional[str], float]:
    label = _most_frequent_label(recent_predictions)
    return label, float(last_prediction_confidence)


def _flatten_frame_features(landmarks: dict) -> Tuple[List[float], bool]:
    return flatten_dynamic_frame(landmarks)


def _predict_from_landmarks(landmarks: dict) -> str:
    global sequence_buffer, hand_presence_buffer, recent_predictions, last_prediction_confidence

    frame_features, has_hand = _flatten_frame_features(landmarks)
    sequence_buffer.append(frame_features)
    hand_presence_buffer.append(1 if has_hand else 0)

    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return "Detecting..."

    if sum(hand_presence_buffer) < MIN_HAND_FRAMES_IN_SEQUENCE:
        sequence_buffer = []
        hand_presence_buffer = []
        return "Detecting..."

    input_tensor = torch.tensor([sequence_buffer], dtype=torch.float32)
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)

    confidence = float(torch.max(probabilities).item())
    predicted = int(torch.argmax(probabilities, dim=1).item())

    sequence_buffer = []
    hand_presence_buffer = []

    last_prediction_confidence = confidence
    if confidence >= PREDICTION_CONFIDENCE_THRESHOLD:
        predicted_label = reverse_map.get(predicted, "Detecting...")
        if predicted_label != "Detecting...":
            recent_predictions.append(predicted_label)
            if len(recent_predictions) > PREDICTION_SMOOTHING_WINDOW:
                recent_predictions = recent_predictions[-PREDICTION_SMOOTHING_WINDOW:]

        smoothed_label = _most_frequent_label(recent_predictions)
        return smoothed_label or "Detecting..."

    return "Detecting..."


def predict_dynamic(frame) -> str:
    landmarks = extract_landmarks(frame)
    return _predict_from_landmarks(landmarks)


def _extract_bbox_pixels(frame, landmarks: dict) -> Optional[Tuple[int, int, int, int]]:
    hands = landmarks.get("hands", [])
    if not hands:
        return None

    hand = hands[0]
    if not hand:
        return None

    h, w = frame.shape[:2]
    xs = [p[0] * w for p in hand]
    ys = [p[1] * h for p in hand]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    if span <= 2:
        return None

    pad = span * 0.2
    x1 = max(0, int(min_x - pad))
    y1 = max(0, int(min_y - pad))
    x2 = min(w - 1, int(max_x + pad))
    y2 = min(h - 1, int(max_y + pad))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def run_realtime_demo() -> None:
    camera_index = int(os.getenv("DYNAMIC_CAMERA_INDEX", "0"))
    process_every_n_frames = max(1, int(os.getenv("DYNAMIC_PROCESS_EVERY_N_FRAMES", "2")))

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Unable to open camera for realtime dynamic prediction.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_idx = 0
    last_landmarks = {"hands": [], "face": []}
    display_label = "Detecting..."
    display_conf = 0.0

    fps_ema = 0.0
    prev_ts = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        process_now = (frame_idx % process_every_n_frames) == 0

        if process_now:
            landmarks = extract_landmarks(frame)
            last_landmarks = landmarks
            display_label = _predict_from_landmarks(landmarks)
            _, display_conf = get_last_prediction_info()

        bbox = _extract_bbox_pixels(frame, last_landmarks)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 120), 2)

        now = time.perf_counter()
        dt = max(now - prev_ts, 1e-6)
        prev_ts = now
        fps = 1.0 / dt
        fps_ema = fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * fps)

        label_text = display_label
        if label_text == "Detecting...":
            info_text = f"Label: {label_text}"
        else:
            info_text = f"Label: {label_text} ({display_conf:.2f})"

        cv2.putText(frame, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps_ema:.1f}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(frame, f"Skip: {process_every_n_frames}", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)

        cv2.imshow("Gestura Dynamic Realtime", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_demo()

