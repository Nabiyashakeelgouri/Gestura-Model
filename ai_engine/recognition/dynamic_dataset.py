import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

MAX_HANDS = 2
HAND_POINTS = 21
FACE_POINTS = 468
SEQUENCE_LENGTH = 30
FACE_ANCHOR_INDICES = (1, 10, 13, 152, 234, 454)
HAND_FEATURE_SIZE = (HAND_POINTS * 3) + 6
FACE_FEATURE_SIZE = len(FACE_ANCHOR_INDICES) * 3
DYNAMIC_INPUT_SIZE = (MAX_HANDS * HAND_FEATURE_SIZE) + FACE_FEATURE_SIZE
DEFAULT_DYNAMIC_DATA_FILE = os.path.join("datasets", "dynamic", "dynamic_data.json")
DEFAULT_DYNAMIC_DATASET_PATH = os.path.join("datasets", "dynamic", "recorded_sequences")
FALLBACK_DYNAMIC_DATASET_PATH = "recorded_sequences"


def _resolve_dataset_source(
    base_path: Optional[str],
) -> Tuple[str, Optional[str]]:
    env_file = os.getenv("DYNAMIC_DATA_FILE")
    if env_file:
        return "file", env_file

    if os.path.isfile(DEFAULT_DYNAMIC_DATA_FILE):
        return "file", DEFAULT_DYNAMIC_DATA_FILE

    if base_path:
        return "dir", base_path

    env_path = os.getenv("DYNAMIC_DATASET_DIR")
    if env_path:
        return "dir", env_path

    if os.path.isdir(DEFAULT_DYNAMIC_DATASET_PATH):
        return "dir", DEFAULT_DYNAMIC_DATASET_PATH

    return "dir", FALLBACK_DYNAMIC_DATASET_PATH


def _collect_sequences_from_dir(base_dir: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(base_dir):
        return []

    nested_base = os.path.join(base_dir, "recorded_sequences")
    effective_base = nested_base if os.path.isdir(nested_base) else base_dir

    sequences: List[Dict[str, Any]] = []
    gestures = sorted(
        [
            item
            for item in os.listdir(effective_base)
            if os.path.isdir(os.path.join(effective_base, item))
        ]
    )

    for gesture in gestures:
        gesture_path = os.path.join(effective_base, gesture)
        for file in sorted(os.listdir(gesture_path)):
            if not file.lower().endswith(".json"):
                continue
            file_path = os.path.join(gesture_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                sequence = json.load(f)
            sequences.append({"label": gesture, "sequence": sequence, "source": file_path})
    return sequences


def _load_sequences_from_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("sequences"), list):
        return payload["sequences"]

    raise ValueError(f"Unsupported dynamic data format in: {file_path}")


def _sort_hands_for_stability(hands: List[List[List[float]]]) -> List[List[List[float]]]:
    def hand_center_x(hand: List[List[float]]) -> float:
        if not hand:
            return 0.0
        return sum(point[0] for point in hand) / len(hand)

    return sorted(hands, key=hand_center_x)


def _safe_div(value: float, denom: float) -> float:
    return value / denom if abs(denom) > 1e-6 else 0.0


def _relative_point(point: List[float], origin: List[float], scale: float) -> List[float]:
    return [
        _safe_div(point[0] - origin[0], scale),
        _safe_div(point[1] - origin[1], scale),
        _safe_div(point[2] - origin[2], scale),
    ]


def flatten_dynamic_frame(frame: Dict[str, Any]) -> Tuple[List[float], bool]:
    frame_features: List[float] = []

    raw_hands = frame.get("hands", [])
    hands = _sort_hands_for_stability(raw_hands)
    has_hand = len(hands) > 0

    faces = frame.get("face", [])
    face = faces[0] if faces and len(faces[0]) == FACE_POINTS else []
    nose = face[1] if face else [0.0, 0.0, 0.0]
    left_face = face[234] if face else [0.0, 0.0, 0.0]
    right_face = face[454] if face else [1.0, 0.0, 0.0]
    face_scale = max(
        math.dist((left_face[0], left_face[1]), (right_face[0], right_face[1])) if face else 0.0,
        1e-3,
    )

    for i in range(MAX_HANDS):
        if i < len(hands) and len(hands[i]) == HAND_POINTS:
            hand = hands[i]
            wrist = hand[0]
            xs = [point[0] for point in hand]
            ys = [point[1] for point in hand]
            scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)
            center = [sum(xs) / len(xs), sum(ys) / len(ys), sum(point[2] for point in hand) / len(hand)]

            frame_features.extend(_relative_point(wrist, nose, face_scale))
            frame_features.extend(_relative_point(center, nose, face_scale))
            for point in hand:
                frame_features.extend(_relative_point(point, wrist, scale))
        else:
            frame_features.extend([0.0] * HAND_FEATURE_SIZE)

    if face:
        for idx in FACE_ANCHOR_INDICES:
            frame_features.extend(_relative_point(face[idx], nose, face_scale))
    else:
        frame_features.extend([0.0] * FACE_FEATURE_SIZE)

    return frame_features, has_hand


def _fit_sequence_length(sequence: List[List[float]], target_len: int) -> Optional[List[List[float]]]:
    if not sequence:
        return None

    if len(sequence) >= target_len:
        return sequence[:target_len]

    padded = list(sequence)
    pad_frame = sequence[-1]
    while len(padded) < target_len:
        padded.append(pad_frame)
    return padded


def load_dynamic_dataset(
    base_path: Optional[str] = None,
    sequence_length: int = SEQUENCE_LENGTH,
    min_hand_frames_ratio: float = 0.4,
):
    X: List[List[List[float]]] = []
    y: List[int] = []
    label_map: Dict[str, int] = {}

    source_type, source_value = _resolve_dataset_source(base_path)
    if source_value is None:
        raise FileNotFoundError("Could not resolve dynamic dataset source.")

    if source_type == "file":
        if not os.path.isfile(source_value):
            raise FileNotFoundError(f"Dynamic data file not found: {source_value}")
        raw_sequences = _load_sequences_from_file(source_value)
    else:
        raw_sequences = _collect_sequences_from_dir(source_value)
        if not raw_sequences:
            raise FileNotFoundError(f"Dynamic dataset path not found or empty: {source_value}")

    labels = sorted(
        {
            str(item.get("label", "")).strip()
            for item in raw_sequences
            if str(item.get("label", "")).strip()
        }
    )

    for idx, label in enumerate(labels):
        label_map[label] = idx

    for item in raw_sequences:
        gesture = str(item.get("label", "")).strip()
        sequence = item.get("sequence", [])
        if gesture not in label_map or not isinstance(sequence, list):
            continue

        flattened_sequence: List[List[float]] = []
        hand_frames = 0

        for frame in sequence:
            frame_features, has_hand = flatten_dynamic_frame(frame)
            flattened_sequence.append(frame_features)
            if has_hand:
                hand_frames += 1

        fitted = _fit_sequence_length(flattened_sequence, sequence_length)
        if fitted is None:
            continue

        hand_ratio = hand_frames / max(len(sequence), 1)
        if hand_ratio < min_hand_frames_ratio:
            continue

        X.append(fitted)
        y.append(label_map[gesture])

    if not X:
        raise ValueError(
            f"No valid dynamic training samples found. Check dataset quality at: {source_value}"
        )

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y), label_map
