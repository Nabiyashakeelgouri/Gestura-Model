import os
import time
import math
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from ai_engine.nlp.sentence_builder import correct_sentence
from ai_engine.preprocessing.landmark_extractor import extract_landmarks
from ai_engine.recognition.dynamic_dataset import FACE_POINTS, HAND_POINTS, MAX_HANDS, flatten_dynamic_frame
from ai_engine.recognition.dynamic_model import DynamicGestureModel, load_dynamic_label_map
from ai_engine.recognition.static_model import STATIC_FEATURE_SIZE, StaticGestureModel, build_static_feature_vector


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DATASET_DIR = ROOT_DIR / "datasets" / "static" / "asl_alphabet"
STATIC_MODEL_PATH = ROOT_DIR / "models" / "static_cnn_model.pth"
STATIC_CLASS_MAP_PATH = ROOT_DIR / "models" / "static_class_map.json"
DYNAMIC_MODEL_PATH = ROOT_DIR / "models" / "dynamic_model.pth"
LABEL_MAP_PATH = ROOT_DIR / "models" / "dynamic_label_map.json"
RECORDED_SEQUENCES_DIR = ROOT_DIR / "recorded_sequences"

SEQUENCE_LENGTH = 30
STATIC_CONFIDENCE_THRESHOLD = float(os.getenv("STATIC_CONFIDENCE_THRESHOLD", "0.65"))
STATIC_MIN_MARGIN = float(os.getenv("STATIC_MIN_MARGIN", "0.08"))
STATIC_STABLE_FRAMES = int(os.getenv("STATIC_STABLE_FRAMES", "2"))
STATIC_TEMPORAL_WINDOW = int(os.getenv("STATIC_TEMPORAL_WINDOW", "10"))
STATIC_TEMPORAL_MIN_VOTES = int(os.getenv("STATIC_TEMPORAL_MIN_VOTES", "3"))
STATIC_PREDICTION_WINDOW = int(os.getenv("STATIC_PREDICTION_WINDOW", "10"))
STATIC_DISPLAY_CONFIDENCE_THRESHOLD = float(os.getenv("STATIC_DISPLAY_CONFIDENCE_THRESHOLD", "0.75"))
DYNAMIC_CONFIDENCE_THRESHOLD = float(os.getenv("DYNAMIC_CONFIDENCE_THRESHOLD", "0.45"))
DYNAMIC_WINDOW_STRIDE = int(os.getenv("DYNAMIC_WINDOW_STRIDE", "10"))
DYNAMIC_STABLE_WINDOWS = int(os.getenv("DYNAMIC_STABLE_WINDOWS", "2"))
DYNAMIC_VOTE_WINDOW = int(os.getenv("DYNAMIC_VOTE_WINDOW", "3"))
DYNAMIC_VOTE_MIN = int(os.getenv("DYNAMIC_VOTE_MIN", "2"))
DYNAMIC_EMIT_COOLDOWN_SECONDS = float(os.getenv("DYNAMIC_EMIT_COOLDOWN_SECONDS", "1.0"))
VIDEO_FRAME_STRIDE = int(os.getenv("VIDEO_FRAME_STRIDE", "3"))
DYNAMIC_MIN_HAND_FRAMES = int(os.getenv("DYNAMIC_MIN_HAND_FRAMES", "8"))
DYNAMIC_MAX_NO_PRED_WINDOWS = int(os.getenv("DYNAMIC_MAX_NO_PRED_WINDOWS", "3"))
DYNAMIC_MOTION_THRESHOLD = float(os.getenv("DYNAMIC_MOTION_THRESHOLD", "0.018"))
DYNAMIC_SUPPRESS_WHEN_STATIC = os.getenv("DYNAMIC_SUPPRESS_WHEN_STATIC", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DYNAMIC_FORCE_CONFIDENCE_THRESHOLD = float(os.getenv("DYNAMIC_FORCE_CONFIDENCE_THRESHOLD", "0.24"))
DYNAMIC_FORCE_TOP1_MARGIN = float(os.getenv("DYNAMIC_FORCE_TOP1_MARGIN", "0.06"))
DYNAMIC_FORCE_LABELS = {
    label.strip()
    for label in os.getenv("DYNAMIC_FORCE_LABELS", "how,are_you").split(",")
    if label.strip()
}
DYNAMIC_FORCE_STABLE_WINDOWS = int(os.getenv("DYNAMIC_FORCE_STABLE_WINDOWS", "2"))
DYNAMIC_FORCE_LABEL_THRESHOLDS_RAW = os.getenv(
    "DYNAMIC_FORCE_LABEL_THRESHOLDS", "how:0.30,are_you:0.32"
)


def _parse_dynamic_force_thresholds(raw: str) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    if not raw:
        return parsed
    for item in raw.split(","):
        token = item.strip()
        if not token or ":" not in token:
            continue
        label, value = token.split(":", 1)
        label = label.strip()
        if not label:
            continue
        try:
            conf = float(value.strip())
        except ValueError:
            continue
        parsed[label] = max(0.0, min(1.0, conf))
    return parsed


DYNAMIC_FORCE_LABEL_THRESHOLDS = _parse_dynamic_force_thresholds(
    DYNAMIC_FORCE_LABEL_THRESHOLDS_RAW
)


def _dynamic_force_threshold(label: str) -> float:
    return float(DYNAMIC_FORCE_LABEL_THRESHOLDS.get(label, DYNAMIC_FORCE_CONFIDENCE_THRESHOLD))


class StaticCNN(StaticGestureModel):
    def __init__(self, num_classes: int, input_size: int = STATIC_FEATURE_SIZE):
        super().__init__(num_classes=num_classes, input_size=input_size)


class LSTMModel(DynamicGestureModel):
    def __init__(self, num_classes: int, num_layers: int = 2):
        super().__init__(num_classes=num_classes, num_layers=num_layers)


def _resolve_dynamic_labels() -> List[str]:
    label_map = load_dynamic_label_map()
    if label_map:
        ordered = sorted(label_map.items(), key=lambda item: int(item[1]))
        return [str(label) for label, _ in ordered]

    env_labels = os.getenv("GESTURA_DYNAMIC_LABELS", "").strip()
    if env_labels:
        labels = [label.strip() for label in env_labels.split(",") if label.strip()]
        if labels:
            return labels

    default_labels = ["hello", "no", "stop", "thank_u", "yes"]

    if not RECORDED_SEQUENCES_DIR.exists():
        return default_labels

    folder_labels = sorted(
        [path.name for path in RECORDED_SEQUENCES_DIR.iterdir() if path.is_dir()]
    )
    return folder_labels if folder_labels else default_labels


def _load_static_resources() -> Tuple[List[str], Optional[StaticCNN]]:
    classes: List[str] = []
    if STATIC_CLASS_MAP_PATH.exists():
        try:
            payload = json.loads(STATIC_CLASS_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                classes = [str(item) for item in payload if str(item)]
        except Exception:
            classes = []

    if not classes and STATIC_DATASET_DIR.exists():
        classes = sorted([item.name for item in STATIC_DATASET_DIR.iterdir() if item.is_dir()])

    if not classes:
        return [], None

    if not STATIC_MODEL_PATH.exists():
        return classes, None

    try:
        checkpoint = torch.load(STATIC_MODEL_PATH, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            input_size = int(checkpoint.get("input_size", STATIC_FEATURE_SIZE))
        else:
            state_dict = checkpoint
            input_size = STATIC_FEATURE_SIZE
        model = StaticCNN(len(classes), input_size=input_size)
        model.load_state_dict(state_dict)
        model.eval()
        return classes, model
    except Exception as exc:
        print(f"Static model load failed: {exc}")
        return classes, None


def _load_dynamic_model(num_classes: int) -> Optional[nn.Module]:
    if not DYNAMIC_MODEL_PATH.exists():
        return None

    try:
        state_dict = torch.load(DYNAMIC_MODEL_PATH, map_location="cpu")
        model = LSTMModel(num_classes)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as exc:
        print(f"Dynamic model load failed: {exc}")
        return None


STATIC_CLASSES, STATIC_MODEL = _load_static_resources()

DYNAMIC_LABELS = _resolve_dynamic_labels()
DYNAMIC_REVERSE_MAP = {index: label for index, label in enumerate(DYNAMIC_LABELS)}
DYNAMIC_MODEL = _load_dynamic_model(len(DYNAMIC_LABELS))


@dataclass
class SessionState:
    status: str = "standby"
    sequence: List[List[float]] = field(default_factory=list)
    sequence_hand_presence: List[int] = field(default_factory=list)
    sequence_motion_scores: List[float] = field(default_factory=list)
    words: List[str] = field(default_factory=list)
    last_static_label: str = ""
    static_candidate_label: str = ""
    static_candidate_streak: int = 0
    static_recent_labels: List[str] = field(default_factory=list)
    last_dynamic_emit_ts: float = 0.0
    dynamic_recent_labels: List[str] = field(default_factory=list)
    dynamic_candidate_label: str = ""
    dynamic_candidate_streak: int = 0
    checkmark_streak: int = 0
    no_hand_streak: int = 0
    no_dynamic_prediction_windows: int = 0
    prev_hand_center: Optional[Tuple[float, float]] = None
    motion_ema: float = 0.0
    cooldown_until: float = 0.0

    def clear_dynamic_state(self) -> None:
        self.sequence.clear()
        self.sequence_hand_presence.clear()
        self.sequence_motion_scores.clear()
        self.no_hand_streak = 0
        self.checkmark_streak = 0
        self.no_dynamic_prediction_windows = 0
        self.dynamic_recent_labels.clear()
        self.dynamic_candidate_label = ""
        self.dynamic_candidate_streak = 0

    def reset_static_candidate(self) -> None:
        self.static_candidate_label = ""
        self.static_candidate_streak = 0
        self.static_recent_labels.clear()


_session_lock = Lock()
_session_state: Dict[str, SessionState] = defaultdict(SessionState)


def _is_finger_extended(hand: List[List[float]], tip_idx: int, pip_idx: int) -> bool:
    return hand[tip_idx][1] < hand[pip_idx][1]


def _is_finger_extended_margin(
    hand: List[List[float]], tip_idx: int, pip_idx: int, margin: float = 0.015
) -> bool:
    return hand[tip_idx][1] < (hand[pip_idx][1] - margin)


def _is_finger_folded_margin(
    hand: List[List[float]], tip_idx: int, pip_idx: int, margin: float = 0.01
) -> bool:
    return hand[tip_idx][1] > (hand[pip_idx][1] + margin)


def _dist2d(p1: List[float], p2: List[float]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _is_finger_horizontal_extended(
    hand: List[List[float]],
    tip_idx: int,
    mcp_idx: int,
    min_x_delta: float = 0.075,
    max_y_delta: float = 0.11,
) -> bool:
    tip = hand[tip_idx]
    mcp = hand[mcp_idx]
    return abs(tip[0] - mcp[0]) > min_x_delta and abs(tip[1] - mcp[1]) < max_y_delta


def _is_finger_vertical_extended(
    hand: List[List[float]],
    tip_idx: int,
    mcp_idx: int,
    min_y_delta: float = 0.09,
    max_x_delta: float = 0.10,
) -> bool:
    tip = hand[tip_idx]
    mcp = hand[mcp_idx]
    return abs(tip[1] - mcp[1]) > min_y_delta and abs(tip[0] - mcp[0]) < max_x_delta


def _palm_center(hand: List[List[float]]) -> List[float]:
    return [
        (hand[5][0] + hand[9][0] + hand[13][0] + hand[17][0]) / 4.0,
        (hand[5][1] + hand[9][1] + hand[13][1] + hand[17][1]) / 4.0,
    ]


def _is_finger_compact_to_palm(
    hand: List[List[float]],
    tip_idx: int,
    mcp_idx: int,
    palm_center: List[float],
    max_tip_palm: float = 0.18,
    max_tip_mcp: float = 0.16,
) -> bool:
    tip = hand[tip_idx]
    mcp = hand[mcp_idx]
    return _dist2d(tip, palm_center) < max_tip_palm and _dist2d(tip, mcp) < max_tip_mcp


def _estimate_static_rule_label(landmarks: Dict[str, List]) -> Optional[str]:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return None

    hand = hands[0]
    if len(hand) < 21:
        return None

    index_extended = _is_finger_extended_margin(hand, 8, 6)
    middle_extended = _is_finger_extended_margin(hand, 12, 10)
    ring_extended = _is_finger_extended_margin(hand, 16, 14)
    pinky_extended = _is_finger_extended_margin(hand, 20, 18)
    extended_count = int(index_extended) + int(middle_extended) + int(ring_extended) + int(pinky_extended)

    thumb_tip = hand[4]
    wrist = hand[0]
    index_tip = hand[8]
    index_pip = hand[6]
    middle_tip = hand[12]
    middle_pip = hand[10]
    ring_tip = hand[16]
    ring_pip = hand[14]
    pinky_tip = hand[20]
    pinky_pip = hand[18]
    index_mcp = hand[5]
    thumb_index_dist = ((thumb_tip[0] - index_mcp[0]) ** 2 + (thumb_tip[1] - index_mcp[1]) ** 2) ** 0.5
    thumb_middle_dist = _dist2d(thumb_tip, middle_tip)
    thumb_index_tip_dist = _dist2d(thumb_tip, index_tip)
    thumb_wrist_dist = _dist2d(thumb_tip, wrist)
    index_wrist_dist = _dist2d(index_tip, wrist)
    index_middle_tip_dist = _dist2d(index_tip, middle_tip)
    folded_fingers_compact = _dist2d(middle_tip, ring_tip) < 0.09 and _dist2d(ring_tip, pinky_tip) < 0.09

    # ASL-B: four fingers mostly extended, thumb tucked across palm.
    if extended_count >= 3 and thumb_index_dist < 0.10:
        return "B"

    # ASL-F: thumb/index make a loop, middle-ring-pinky extended.
    if (
        thumb_index_tip_dist < 0.065
        and _is_finger_extended_margin(hand, 12, 10, margin=0.015)
        and _is_finger_extended_margin(hand, 16, 14, margin=0.015)
        and _is_finger_extended_margin(hand, 20, 18, margin=0.015)
    ):
        return "F"

    palm_center = _palm_center(hand)
    index_reach = _dist2d(index_tip, palm_center)
    middle_reach = _dist2d(middle_tip, palm_center)
    ring_reach = _dist2d(ring_tip, palm_center)
    pinky_reach = _dist2d(pinky_tip, palm_center)

    # ASL-H: index + middle extended sideways, ring/pinky folded.
    index_horizontal = _is_finger_horizontal_extended(hand, 8, 5)
    middle_horizontal = _is_finger_horizontal_extended(hand, 12, 9)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center)
    index_middle_parallel = abs((index_tip[1] - hand[5][1]) - (middle_tip[1] - hand[9][1])) < 0.08
    if (
        index_horizontal
        and middle_horizontal
        and ring_folded
        and pinky_folded
        and index_middle_parallel
        and index_reach > 0.15
        and middle_reach > 0.15
        and ring_reach < 0.20
        and pinky_reach < 0.20
    ):
        return "H"

    # ASL-G: index extended sideways, middle/ring/pinky folded, thumb near index.
    if (
        index_horizontal
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center)
        and ring_folded
        and pinky_folded
        and thumb_index_tip_dist < 0.12
        and thumb_wrist_dist > 0.10
        and index_reach > 0.16
        and middle_reach < 0.20
        and ring_reach < 0.20
        and pinky_reach < 0.20
    ):
        return "G"

    # ASL-L: index extended + thumb extended sideways, other fingers folded.
    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        index_strong = _is_finger_extended_margin(hand, 8, 6, margin=0.04)
        middle_folded = _is_finger_folded_margin(hand, 12, 10, margin=0.005)
        ring_folded = _is_finger_folded_margin(hand, 16, 14, margin=0.005)
        pinky_folded = _is_finger_folded_margin(hand, 20, 18, margin=0.005)

        vec_index = (index_tip[0] - wrist[0], index_tip[1] - wrist[1])
        vec_thumb = (thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
        norm_i = max((vec_index[0] ** 2 + vec_index[1] ** 2) ** 0.5, 1e-6)
        norm_t = max((vec_thumb[0] ** 2 + vec_thumb[1] ** 2) ** 0.5, 1e-6)
        cos_angle = (vec_index[0] * vec_thumb[0] + vec_index[1] * vec_thumb[1]) / (norm_i * norm_t)
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

        thumb_out = thumb_wrist_dist > 0.17 and thumb_index_dist > 0.13
        index_out = index_wrist_dist > 0.18
        l_shape = 55.0 <= angle_deg <= 125.0

        if index_strong and middle_folded and ring_folded and pinky_folded and thumb_out and index_out and l_shape:
            return "L"

    # ASL-I: pinky vertical, other fingers compact/folded.
    pinky_vertical = _is_finger_vertical_extended(hand, 20, 17, min_y_delta=0.08, max_x_delta=0.13)
    index_compact = _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    thumb_near_palm = _dist2d(thumb_tip, hand[9]) < 0.17 or _dist2d(thumb_tip, hand[5]) < 0.17
    if pinky_vertical and index_compact and middle_compact and ring_compact and thumb_near_palm:
        return "I"

    # ASL-K: index + middle vertical with slight V split, ring/pinky compact,
    # thumb placed between index and middle bases.
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.14)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.14)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    index_middle_split = _dist2d(index_tip, middle_tip) > 0.05
    index_mcp = hand[5]
    middle_mcp = hand[9]
    thumb_between_x = min(index_mcp[0], middle_mcp[0]) - 0.05 <= thumb_tip[0] <= max(index_mcp[0], middle_mcp[0]) + 0.05
    thumb_near_knuckles = _dist2d(thumb_tip, index_mcp) < 0.15 and _dist2d(thumb_tip, middle_mcp) < 0.15
    if (
        index_vertical
        and middle_vertical
        and ring_compact
        and pinky_compact
        and index_middle_split
        and thumb_between_x
        and thumb_near_knuckles
    ):
        return "K"



    # ASL-R: index + middle extended and crossed, ring/pinky folded.
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.16)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.16)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    tips_close = _dist2d(index_tip, middle_tip) < 0.055
    thumb_near_palm = _dist2d(thumb_tip, hand[9]) < 0.18 or _dist2d(thumb_tip, hand[5]) < 0.18
    if index_vertical and middle_vertical and ring_compact and pinky_compact and tips_close and thumb_near_palm:
        return "R"


    # ASL-U: index + middle vertical and close together, ring/pinky folded.
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.16)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.16)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    tips_close_u = _dist2d(index_tip, middle_tip) < 0.070
    thumb_near_palm_u = _dist2d(thumb_tip, hand[5]) < 0.18 or _dist2d(thumb_tip, hand[9]) < 0.18
    if index_vertical and middle_vertical and ring_compact and pinky_compact and tips_close_u and thumb_near_palm_u:
        return "U"

    # ASL-V: index + middle vertical and separated, ring/pinky folded.
    tips_split_v = _dist2d(index_tip, middle_tip) > 0.075
    if index_vertical and middle_vertical and ring_compact and pinky_compact and tips_split_v:
        return "V"

    # ASL-W: index + middle + ring vertical, pinky folded.
    ring_vertical = _is_finger_vertical_extended(hand, 16, 13, min_y_delta=0.07, max_x_delta=0.17)
    pinky_folded_w = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    if index_vertical and middle_vertical and ring_vertical and pinky_folded_w:
        return "W"

    # ASL-X: index hooked (partially folded), others folded, thumb near index base.
    index_hooked = hand[8][1] < hand[6][1] and hand[8][1] > hand[5][1]
    middle_folded_x = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
    ring_folded_x = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
    pinky_folded_x = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
    thumb_near_index_base = _dist2d(thumb_tip, hand[5]) < 0.16
    if index_hooked and middle_folded_x and ring_folded_x and pinky_folded_x and thumb_near_index_base:
        return "X"

    # ASL-Y: thumb + pinky extended, index/middle/ring folded.
    thumb_extended_y = _dist2d(thumb_tip, wrist) > 0.18
    pinky_extended_y = _is_finger_vertical_extended(hand, 20, 17, min_y_delta=0.07, max_x_delta=0.15)
    index_folded_y = _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    middle_folded_y = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    ring_folded_y = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    if thumb_extended_y and pinky_extended_y and index_folded_y and middle_folded_y and ring_folded_y:
        return "Y"

    # ASL-P: like K but downward orientation.
    index_down = (hand[8][1] - hand[5][1]) > 0.07 and abs(hand[8][0] - hand[5][0]) < 0.16
    middle_down = (hand[12][1] - hand[9][1]) > 0.07 and abs(hand[12][0] - hand[9][0]) < 0.16
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    index_middle_split = _dist2d(index_tip, middle_tip) > 0.05
    index_mcp = hand[5]
    middle_mcp = hand[9]
    thumb_between_x = min(index_mcp[0], middle_mcp[0]) - 0.07 <= thumb_tip[0] <= max(index_mcp[0], middle_mcp[0]) + 0.07
    thumb_near_knuckles = _dist2d(thumb_tip, index_mcp) < 0.18 and _dist2d(thumb_tip, middle_mcp) < 0.18
    if (
        index_down
        and middle_down
        and ring_compact
        and pinky_compact
        and index_middle_split
        and thumb_between_x
        and thumb_near_knuckles
    ):
        return "P"

    # ASL-Q: like G but downward orientation.
    index_down = (hand[8][1] - hand[5][1]) > 0.08 and abs(hand[8][0] - hand[5][0]) < 0.16
    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    thumb_index_close = _dist2d(thumb_tip, index_tip) < 0.13
    if index_down and middle_compact and ring_compact and pinky_compact and thumb_index_close:
        return "Q"

    # ASL-D: index extended, others folded, thumb near middle/index base.
    index_vertical = _is_finger_vertical_extended(hand, 8, 5)
    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    if (
        index_extended
        and index_vertical
        and not index_horizontal
        and middle_compact
        and ring_compact
        and pinky_compact
        and (thumb_middle_dist < 0.14 or _dist2d(thumb_tip, hand[5]) < 0.13)
        and index_middle_tip_dist > 0.12
    ):
        return "D"

    # ASL-M: thumb tucked under index/middle/ring; closed compact fist shape.
    thumb_under_three = thumb_tip[1] > max(index_tip[1], middle_tip[1], ring_tip[1]) + 0.01
    thumb_near_knuckles = (
        _dist2d(thumb_tip, hand[5]) < 0.16
        and _dist2d(thumb_tip, hand[9]) < 0.16
        and _dist2d(thumb_tip, hand[13]) < 0.17
    )
    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.20, max_tip_mcp=0.18)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.20, max_tip_mcp=0.18)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.20, max_tip_mcp=0.18)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
    )
    thumb_crossing_e = _dist2d(thumb_tip, index_tip) < 0.11 and _dist2d(thumb_tip, middle_tip) < 0.12
    if fist_compact and thumb_under_three and thumb_near_knuckles and not thumb_crossing_e:
        return "M"

    # ASL-N: thumb tucked under index/middle (not under ring), compact fist.
    thumb_under_two = thumb_tip[1] > max(index_tip[1], middle_tip[1]) + 0.01
    thumb_not_under_ring = thumb_tip[1] <= ring_tip[1] + 0.015
    thumb_near_first_two = _dist2d(thumb_tip, hand[5]) < 0.16 and _dist2d(thumb_tip, hand[9]) < 0.16
    thumb_far_from_ring_base = _dist2d(thumb_tip, hand[13]) > 0.10
    if (
        fist_compact
        and thumb_under_two
        and thumb_not_under_ring
        and thumb_near_first_two
        and thumb_far_from_ring_base
        and not thumb_crossing_e
    ):
        return "N"

    # ASL-O: all fingertips curved into an "O" with thumb near index/middle tips.
    fingertip_cluster = (
        _dist2d(index_tip, middle_tip) < 0.12
        and _dist2d(middle_tip, ring_tip) < 0.12
        and _dist2d(ring_tip, pinky_tip) < 0.13
    )
    thumb_forms_o = _dist2d(thumb_tip, index_tip) < 0.12 and _dist2d(thumb_tip, middle_tip) < 0.14
    tips_not_fully_crushed = (
        0.12 < _dist2d(index_tip, palm_center) < 0.30
        and 0.12 < _dist2d(middle_tip, palm_center) < 0.30
    )
    if fingertip_cluster and thumb_forms_o and tips_not_fully_crushed:
        return "O"

    # ASL-S: tight fist with thumb crossing in front (not tucked under fingers).
    fist_compact_s = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    )
    thumb_cross_front = _dist2d(thumb_tip, index_tip) < 0.10 and _dist2d(thumb_tip, middle_tip) < 0.12
    thumb_not_under = thumb_tip[1] <= middle_tip[1] + 0.03
    if fist_compact_s and thumb_cross_front and thumb_not_under:
        return "S"

    # ASL-T: fist with thumb tucked between index and middle (more centered than S).
    thumb_between_x = min(hand[5][0], hand[9][0]) - 0.06 <= thumb_tip[0] <= max(hand[5][0], hand[9][0]) + 0.06
    thumb_between_knuckles = _dist2d(thumb_tip, hand[5]) < 0.15 and _dist2d(thumb_tip, hand[9]) < 0.15
    thumb_not_under_ring = thumb_tip[1] <= ring_tip[1] + 0.02
    thumb_not_far_front = _dist2d(thumb_tip, index_tip) > 0.06
    if fist_compact_s and thumb_between_x and thumb_between_knuckles and thumb_not_under_ring and thumb_not_far_front:
        return "T"

    # ASL-A: all fingers curled, thumb visible by index base, compact fist.
    if (
        extended_count == 0
        and 0.08 <= thumb_index_dist <= 0.20
        and folded_fingers_compact
    ):
        return "A"

    # ASL-E: fingertips curled tightly toward palm, thumb crossing front.
    tip_cluster = (
        _dist2d(index_tip, palm_center) < 0.16
        and _dist2d(middle_tip, palm_center) < 0.16
        and _dist2d(ring_tip, palm_center) < 0.16
        and _dist2d(pinky_tip, palm_center) < 0.16
    )
    folded_all = (
        _is_finger_folded_margin(hand, 8, 6, margin=0.002)
        and _is_finger_folded_margin(hand, 12, 10, margin=0.002)
        and _is_finger_folded_margin(hand, 16, 14, margin=0.002)
        and _is_finger_folded_margin(hand, 20, 18, margin=0.002)
    )
    thumb_crossing = _dist2d(thumb_tip, index_tip) < 0.12 and _dist2d(thumb_tip, middle_tip) < 0.13
    if tip_cluster and folded_all and thumb_crossing:
        return "E"

    return None


def _is_strict_l_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    wrist = hand[0]
    thumb_tip = hand[4]
    index_tip = hand[8]

    index_strong = _is_finger_extended_margin(hand, 8, 6, margin=0.04)
    middle_folded = _is_finger_folded_margin(hand, 12, 10, margin=0.005)
    ring_folded = _is_finger_folded_margin(hand, 16, 14, margin=0.005)
    pinky_folded = _is_finger_folded_margin(hand, 20, 18, margin=0.005)

    vec_index = (index_tip[0] - wrist[0], index_tip[1] - wrist[1])
    vec_thumb = (thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
    norm_i = max((vec_index[0] ** 2 + vec_index[1] ** 2) ** 0.5, 1e-6)
    norm_t = max((vec_thumb[0] ** 2 + vec_thumb[1] ** 2) ** 0.5, 1e-6)
    cos_angle = (vec_index[0] * vec_thumb[0] + vec_index[1] * vec_thumb[1]) / (norm_i * norm_t)
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

    thumb_index_dist = _dist2d(thumb_tip, hand[5])
    thumb_wrist_dist = _dist2d(thumb_tip, wrist)
    index_wrist_dist = _dist2d(index_tip, wrist)
    thumb_out = thumb_wrist_dist > 0.17 and thumb_index_dist > 0.13
    index_out = index_wrist_dist > 0.18
    l_shape = 55.0 <= angle_deg <= 125.0

    return (
        index_strong
        and middle_folded
        and ring_folded
        and pinky_folded
        and thumb_out
        and index_out
        and l_shape
    )


def _is_strict_f_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    thumb_index_tip_dist = _dist2d(thumb_tip, index_tip)
    middle_extended = _is_finger_extended_margin(hand, 12, 10, margin=0.015)
    ring_extended = _is_finger_extended_margin(hand, 16, 14, margin=0.015)
    pinky_extended = _is_finger_extended_margin(hand, 20, 18, margin=0.015)

    # Keep separation from B by requiring clear loop and compact index-thumb cluster.
    loop_shape = thumb_index_tip_dist < 0.065
    compact_loop = (
        _dist2d(thumb_tip, hand[6]) < 0.11
        and _dist2d(index_tip, hand[6]) < 0.10
    )

    return loop_shape and compact_loop and middle_extended and ring_extended and pinky_extended


def _is_strict_h_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    index_horizontal = _is_finger_horizontal_extended(hand, 8, 5, min_x_delta=0.07, max_y_delta=0.11)
    middle_horizontal = _is_finger_horizontal_extended(hand, 12, 9, min_x_delta=0.07, max_y_delta=0.11)
    parallel = abs((hand[8][1] - hand[5][1]) - (hand[12][1] - hand[9][1])) < 0.07
    palm_center = _palm_center(hand)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center)
    index_reach = _dist2d(hand[8], palm_center)
    middle_reach = _dist2d(hand[12], palm_center)
    ring_reach = _dist2d(hand[16], palm_center)
    pinky_reach = _dist2d(hand[20], palm_center)

    return (
        index_horizontal
        and middle_horizontal
        and ring_folded
        and pinky_folded
        and parallel
        and index_reach > 0.15
        and middle_reach > 0.15
        and ring_reach < 0.20
        and pinky_reach < 0.20
    )

def _is_loose_h_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    index_horizontal = _is_finger_horizontal_extended(hand, 8, 5, min_x_delta=0.05, max_y_delta=0.16)
    middle_horizontal = _is_finger_horizontal_extended(hand, 12, 9, min_x_delta=0.05, max_y_delta=0.16)
    parallel = abs((hand[8][1] - hand[5][1]) - (hand[12][1] - hand[9][1])) < 0.12
    palm_center = _palm_center(hand)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.26, max_tip_mcp=0.22)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.26, max_tip_mcp=0.22)
    index_reach = _dist2d(hand[8], palm_center)
    middle_reach = _dist2d(hand[12], palm_center)

    return (
        index_horizontal
        and middle_horizontal
        and parallel
        and ring_folded
        and pinky_folded
        and index_reach > 0.12
        and middle_reach > 0.12
    )


def _is_strict_g_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    index_horizontal = _is_finger_horizontal_extended(hand, 8, 5, min_x_delta=0.085, max_y_delta=0.10)
    palm_center = _palm_center(hand)
    middle_folded = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.18, max_tip_mcp=0.16)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.18, max_tip_mcp=0.16)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.19, max_tip_mcp=0.17)
    thumb_index_dist = _dist2d(hand[4], hand[8])
    thumb_index_close = thumb_index_dist < 0.095
    thumb_index_y_close = abs(hand[4][1] - hand[8][1]) < 0.075
    thumb_visible = _dist2d(hand[4], hand[0]) > 0.10
    index_reach = _dist2d(hand[8], palm_center)
    middle_reach = _dist2d(hand[12], palm_center)
    ring_reach = _dist2d(hand[16], palm_center)
    pinky_reach = _dist2d(hand[20], palm_center)
    index_middle_gap = _dist2d(hand[8], hand[12]) > 0.11

    return (
        index_horizontal
        and middle_folded
        and ring_folded
        and pinky_folded
        and thumb_index_close
        and thumb_index_y_close
        and thumb_visible
        and index_reach > 0.17
        and middle_reach < 0.17
        and ring_reach < 0.17
        and pinky_reach < 0.18
        and index_middle_gap
    )


def _is_loose_g_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_horizontal = _is_finger_horizontal_extended(hand, 8, 5, min_x_delta=0.07, max_y_delta=0.12)
    index_not_vertical = not _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.12)
    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.20, max_tip_mcp=0.18)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    thumb_visible = _dist2d(hand[4], hand[0]) > 0.07
    thumb_index_dist = _dist2d(hand[4], hand[8])
    thumb_index_y_close = abs(hand[4][1] - hand[8][1]) < 0.09
    index_reach = _dist2d(hand[8], palm_center)
    middle_reach = _dist2d(hand[12], palm_center)
    ring_reach = _dist2d(hand[16], palm_center)
    pinky_reach = _dist2d(hand[20], palm_center)

    return (
        index_horizontal
        and index_not_vertical
        and middle_compact
        and ring_folded
        and pinky_folded
        and thumb_visible
        and thumb_index_dist < 0.11
        and thumb_index_y_close
        and index_reach > 0.14
        and middle_reach < 0.19
        and ring_reach < 0.19
        and pinky_reach < 0.20
    )


def _is_strict_i_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False
    palm_center = _palm_center(hand)
    pinky_vertical = _is_finger_vertical_extended(hand, 20, 17, min_y_delta=0.08, max_x_delta=0.13)
    index_compact = _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.21, max_tip_mcp=0.18)
    thumb_near_palm = _dist2d(hand[4], hand[9]) < 0.17 or _dist2d(hand[4], hand[5]) < 0.17
    return pinky_vertical and index_compact and middle_compact and ring_compact and thumb_near_palm


def _is_strict_k_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.15)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.15)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    index_middle_split = _dist2d(hand[8], hand[12]) > 0.05
    index_mcp = hand[5]
    middle_mcp = hand[9]
    thumb_tip = hand[4]
    thumb_between_x = min(index_mcp[0], middle_mcp[0]) - 0.06 <= thumb_tip[0] <= max(index_mcp[0], middle_mcp[0]) + 0.06
    thumb_near_knuckles = _dist2d(thumb_tip, index_mcp) < 0.16 and _dist2d(thumb_tip, middle_mcp) < 0.16

    return (
        index_vertical
        and middle_vertical
        and ring_compact
        and pinky_compact
        and index_middle_split
        and thumb_between_x
        and thumb_near_knuckles
    )



def _is_loose_k_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_tip = hand[8]
    middle_tip = hand[12]
    index_mcp = hand[5]
    middle_mcp = hand[9]
    thumb_tip = hand[4]

    # Relaxed K: keep key structure while tolerating noisy landmark jitter.
    index_up = (index_mcp[1] - index_tip[1]) > 0.03
    middle_up = (middle_mcp[1] - middle_tip[1]) > 0.03
    index_verticalish = abs(index_tip[0] - index_mcp[0]) < 0.18
    middle_verticalish = abs(middle_tip[0] - middle_mcp[0]) < 0.18
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    index_middle_split = _dist2d(index_tip, middle_tip) > 0.035
    thumb_between_x = min(index_mcp[0], middle_mcp[0]) - 0.08 <= thumb_tip[0] <= max(index_mcp[0], middle_mcp[0]) + 0.08
    thumb_near_knuckles = _dist2d(thumb_tip, index_mcp) < 0.19 and _dist2d(thumb_tip, middle_mcp) < 0.19

    return (
        index_up
        and middle_up
        and index_verticalish
        and middle_verticalish
        and ring_compact
        and pinky_compact
        and index_middle_split
        and thumb_between_x
        and thumb_near_knuckles
    )


def _is_strict_a_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]

    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.23, max_tip_mcp=0.21)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.23, max_tip_mcp=0.21)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    )

    thumb_index_base_dist = _dist2d(thumb_tip, hand[5])
    thumb_visible_by_index = 0.07 <= thumb_index_base_dist <= 0.22
    thumb_not_under_fingers = thumb_tip[1] <= hand[12][1] + 0.04

    # Avoid P/Q style downward two-finger geometry.
    p_like_down = (
        (hand[8][1] - hand[5][1]) > 0.05
        and (hand[12][1] - hand[9][1]) > 0.05
        and _dist2d(index_tip, middle_tip) > 0.045
    )

    # Avoid E-style front thumb crossing being treated as A.
    e_like_thumb_cross = _dist2d(thumb_tip, index_tip) < 0.15 and _dist2d(thumb_tip, middle_tip) < 0.16

    return (
        fist_compact
        and thumb_visible_by_index
        and thumb_not_under_fingers
        and (not p_like_down)
        and (not e_like_thumb_cross)
    )


def _is_strict_d_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)

    index_extended = _is_finger_extended_margin(hand, 8, 6, margin=0.03)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.09, max_x_delta=0.14)

    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)

    # Keep D separate from U/V/W: middle tip should stay near palm and away from index tip.
    middle_not_vertical = not _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.07, max_x_delta=0.18)
    index_middle_sep = _dist2d(hand[8], hand[12]) > 0.11

    thumb_tip = hand[4]
    thumb_near_base = _dist2d(thumb_tip, hand[5]) < 0.15 or _dist2d(thumb_tip, hand[9]) < 0.15

    return (
        index_extended
        and index_vertical
        and middle_compact
        and ring_compact
        and pinky_compact
        and middle_not_vertical
        and index_middle_sep
        and thumb_near_base
    )

def _is_strict_r_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.16)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.16)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    tips_close = _dist2d(hand[8], hand[12]) < 0.055
    thumb_near_palm = _dist2d(hand[4], hand[9]) < 0.18 or _dist2d(hand[4], hand[5]) < 0.18
    return index_vertical and middle_vertical and ring_compact and pinky_compact and tips_close and thumb_near_palm


def _is_strict_p_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_down = (hand[8][1] - hand[5][1]) > 0.07 and abs(hand[8][0] - hand[5][0]) < 0.16
    middle_down = (hand[12][1] - hand[9][1]) > 0.07 and abs(hand[12][0] - hand[9][0]) < 0.16
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    index_middle_split = _dist2d(hand[8], hand[12]) > 0.05
    index_mcp = hand[5]
    middle_mcp = hand[9]
    thumb_tip = hand[4]
    thumb_between_x = min(index_mcp[0], middle_mcp[0]) - 0.07 <= thumb_tip[0] <= max(index_mcp[0], middle_mcp[0]) + 0.07
    thumb_near_knuckles = _dist2d(thumb_tip, index_mcp) < 0.18 and _dist2d(thumb_tip, middle_mcp) < 0.18
    return (
        index_down
        and middle_down
        and ring_compact
        and pinky_compact
        and index_middle_split
        and thumb_between_x
        and thumb_near_knuckles
    )


def _is_strict_q_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_down = (hand[8][1] - hand[5][1]) > 0.08 and abs(hand[8][0] - hand[5][0]) < 0.16
    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.23, max_tip_mcp=0.20)
    thumb_index_close = _dist2d(hand[4], hand[8]) < 0.13
    return index_down and middle_compact and ring_compact and pinky_compact and thumb_index_close



def _is_loose_r_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.06, max_x_delta=0.18)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.06, max_x_delta=0.18)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    tips_close = _dist2d(hand[8], hand[12]) < 0.075
    return index_vertical and middle_vertical and ring_compact and pinky_compact and tips_close


def _is_loose_p_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_down = (hand[8][1] - hand[5][1]) > 0.05 and abs(hand[8][0] - hand[5][0]) < 0.20
    middle_down = (hand[12][1] - hand[9][1]) > 0.05 and abs(hand[12][0] - hand[9][0]) < 0.20
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.25, max_tip_mcp=0.22)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.25, max_tip_mcp=0.22)
    index_middle_split = _dist2d(hand[8], hand[12]) > 0.035
    return index_down and middle_down and ring_compact and pinky_compact and index_middle_split


def _is_loose_q_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_down = (hand[8][1] - hand[5][1]) > 0.06 and abs(hand[8][0] - hand[5][0]) < 0.20
    middle_compact = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.25, max_tip_mcp=0.22)
    ring_compact = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.25, max_tip_mcp=0.22)
    pinky_compact = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.25, max_tip_mcp=0.22)
    thumb_index_close = _dist2d(hand[4], hand[8]) < 0.16
    return index_down and middle_compact and ring_compact and pinky_compact and thumb_index_close



def _is_strict_o_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    fingertip_cluster = (
        _dist2d(index_tip, middle_tip) < 0.12
        and _dist2d(middle_tip, ring_tip) < 0.12
        and _dist2d(ring_tip, pinky_tip) < 0.13
    )
    thumb_forms_o = _dist2d(thumb_tip, index_tip) < 0.12 and _dist2d(thumb_tip, middle_tip) < 0.14
    tips_not_fully_crushed = (
        0.12 < _dist2d(index_tip, palm_center) < 0.30
        and 0.12 < _dist2d(middle_tip, palm_center) < 0.30
    )
    return fingertip_cluster and thumb_forms_o and tips_not_fully_crushed


def _is_loose_o_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    fingertip_cluster = (
        _dist2d(index_tip, middle_tip) < 0.14
        and _dist2d(middle_tip, ring_tip) < 0.14
        and _dist2d(ring_tip, pinky_tip) < 0.15
    )
    thumb_forms_o = _dist2d(thumb_tip, index_tip) < 0.14 and _dist2d(thumb_tip, middle_tip) < 0.16
    return fingertip_cluster and thumb_forms_o and _dist2d(index_tip, palm_center) > 0.10


def _is_strict_s_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    )
    thumb_cross_front = _dist2d(thumb_tip, hand[8]) < 0.10 and _dist2d(thumb_tip, hand[12]) < 0.12
    thumb_not_under = thumb_tip[1] <= hand[12][1] + 0.03
    return fist_compact and thumb_cross_front and thumb_not_under


def _is_loose_s_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    )
    thumb_frontish = _dist2d(thumb_tip, hand[8]) < 0.13 and _dist2d(thumb_tip, hand[12]) < 0.15
    return fist_compact and thumb_frontish


def _is_strict_t_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    )
    thumb_between_x = min(hand[5][0], hand[9][0]) - 0.06 <= thumb_tip[0] <= max(hand[5][0], hand[9][0]) + 0.06
    thumb_between_knuckles = _dist2d(thumb_tip, hand[5]) < 0.15 and _dist2d(thumb_tip, hand[9]) < 0.15
    thumb_not_under_ring = thumb_tip[1] <= hand[16][1] + 0.02
    thumb_not_far_front = _dist2d(thumb_tip, hand[8]) > 0.06
    return fist_compact and thumb_between_x and thumb_between_knuckles and thumb_not_under_ring and thumb_not_far_front


def _is_loose_t_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    )
    thumb_between_x = min(hand[5][0], hand[9][0]) - 0.08 <= thumb_tip[0] <= max(hand[5][0], hand[9][0]) + 0.08
    thumb_near_index_middle = _dist2d(thumb_tip, hand[5]) < 0.18 and _dist2d(thumb_tip, hand[9]) < 0.18
    return fist_compact and thumb_between_x and thumb_near_index_middle





def _is_strict_u_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.16)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.16)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    tips_close = _dist2d(hand[8], hand[12]) < 0.070
    return index_vertical and middle_vertical and ring_folded and pinky_folded and tips_close


def _is_loose_u_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.06, max_x_delta=0.19)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.06, max_x_delta=0.19)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    tips_close = _dist2d(hand[8], hand[12]) < 0.085
    return index_vertical and middle_vertical and ring_folded and pinky_folded and tips_close


def _is_strict_v_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.16)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.16)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    tips_split = _dist2d(hand[8], hand[12]) > 0.075
    return index_vertical and middle_vertical and ring_folded and pinky_folded and tips_split


def _is_loose_v_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.06, max_x_delta=0.19)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.06, max_x_delta=0.19)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    tips_split = _dist2d(hand[8], hand[12]) > 0.060
    return index_vertical and middle_vertical and ring_folded and pinky_folded and tips_split


def _is_strict_w_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.08, max_x_delta=0.17)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.08, max_x_delta=0.17)
    ring_vertical = _is_finger_vertical_extended(hand, 16, 13, min_y_delta=0.07, max_x_delta=0.17)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    return index_vertical and middle_vertical and ring_vertical and pinky_folded


def _is_loose_w_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_vertical = _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.06, max_x_delta=0.20)
    middle_vertical = _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.06, max_x_delta=0.20)
    ring_vertical = _is_finger_vertical_extended(hand, 16, 13, min_y_delta=0.05, max_x_delta=0.20)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.26, max_tip_mcp=0.23)
    return index_vertical and middle_vertical and ring_vertical and pinky_folded


def _is_strict_x_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_hooked = hand[8][1] < hand[6][1] and hand[8][1] > hand[5][1]
    middle_folded = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
    thumb_near_index_base = _dist2d(hand[4], hand[5]) < 0.16
    return index_hooked and middle_folded and ring_folded and pinky_folded and thumb_near_index_base


def _is_loose_x_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    index_hooked = hand[8][1] < hand[7][1] and hand[8][1] > hand[5][1]
    middle_folded = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    pinky_folded = _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    thumb_near_index_base = _dist2d(hand[4], hand[5]) < 0.19
    return index_hooked and middle_folded and ring_folded and pinky_folded and thumb_near_index_base


def _is_strict_y_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_extended = _dist2d(hand[4], hand[0]) > 0.18
    pinky_extended = _is_finger_vertical_extended(hand, 20, 17, min_y_delta=0.07, max_x_delta=0.15)
    index_folded = _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    middle_folded = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.19)
    return thumb_extended and pinky_extended and index_folded and middle_folded and ring_folded


def _is_loose_y_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_extended = _dist2d(hand[4], hand[0]) > 0.15
    pinky_extended = _is_finger_vertical_extended(hand, 20, 17, min_y_delta=0.05, max_x_delta=0.18)
    index_folded = _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    middle_folded = _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    ring_folded = _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.21)
    return thumb_extended and pinky_extended and index_folded and middle_folded and ring_folded


def _is_strict_c_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    wrist = hand[0]
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]
    index_mcp = hand[5]

    index_curve = 0.09 < _dist2d(index_tip, palm_center) < 0.40
    middle_curve = 0.09 < _dist2d(middle_tip, palm_center) < 0.40
    ring_curve = 0.08 < _dist2d(ring_tip, palm_center) < 0.38
    pinky_curve = 0.07 < _dist2d(pinky_tip, palm_center) < 0.36

    thumb_index_gap = _dist2d(thumb_tip, index_tip)
    c_gap_ok = 0.10 < thumb_index_gap < 0.34
    arc_span_ok = _dist2d(index_tip, pinky_tip) > 0.10
    thumb_not_looping = _dist2d(thumb_tip, index_mcp) > 0.06

    index_horizontal = _is_finger_horizontal_extended(hand, 8, 5, min_x_delta=0.09, max_y_delta=0.12)
    middle_horizontal = _is_finger_horizontal_extended(hand, 12, 9, min_x_delta=0.08, max_y_delta=0.12)
    g_like = index_horizontal and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.20, max_tip_mcp=0.18) and thumb_index_gap < 0.12
    h_like = index_horizontal and middle_horizontal and _dist2d(index_tip, middle_tip) > 0.07
    w_like = (
        _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.07, max_x_delta=0.18)
        and _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.07, max_x_delta=0.18)
        and _is_finger_vertical_extended(hand, 16, 13, min_y_delta=0.06, max_x_delta=0.18)
    )

    compact_cluster = (
        _dist2d(index_tip, middle_tip) < 0.10
        and _dist2d(middle_tip, ring_tip) < 0.10
        and _dist2d(ring_tip, pinky_tip) < 0.11
        and thumb_index_gap < 0.11
    )
    a_like = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.18, max_tip_mcp=0.17)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.18, max_tip_mcp=0.17)
        and thumb_index_gap < 0.11
    )

    thumb_far = _dist2d(thumb_tip, wrist) > 0.22
    pinky_up = _is_finger_vertical_extended(hand, 20, 17, min_y_delta=0.09, max_x_delta=0.14)
    y_like = thumb_far and pinky_up

    return (
        index_curve
        and middle_curve
        and ring_curve
        and pinky_curve
        and c_gap_ok
        and arc_span_ok
        and thumb_not_looping
        and (not y_like)
        and (not g_like)
        and (not h_like)
        and (not compact_cluster)
        and (not a_like)
    )


def _is_loose_c_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    wrist = hand[0]
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]
    index_mcp = hand[5]

    index_curve = 0.07 < _dist2d(index_tip, palm_center) < 0.45
    middle_curve = 0.07 < _dist2d(middle_tip, palm_center) < 0.45
    ring_curve = 0.06 < _dist2d(ring_tip, palm_center) < 0.42
    pinky_curve = 0.05 < _dist2d(pinky_tip, palm_center) < 0.40

    thumb_index_gap = _dist2d(thumb_tip, index_tip)
    c_gap_ok = 0.08 < thumb_index_gap < 0.38
    thumb_not_looping = _dist2d(thumb_tip, index_mcp) > 0.04
    arc_span_ok = _dist2d(index_tip, pinky_tip) > 0.08

    index_horizontal = _is_finger_horizontal_extended(hand, 8, 5, min_x_delta=0.10, max_y_delta=0.10)
    middle_horizontal = _is_finger_horizontal_extended(hand, 12, 9, min_x_delta=0.09, max_y_delta=0.10)
    g_like = index_horizontal and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.18, max_tip_mcp=0.16) and thumb_index_gap < 0.10
    h_like = index_horizontal and middle_horizontal and _dist2d(index_tip, middle_tip) > 0.09
    w_like = (
        _is_finger_vertical_extended(hand, 8, 5, min_y_delta=0.06, max_x_delta=0.20)
        and _is_finger_vertical_extended(hand, 12, 9, min_y_delta=0.06, max_x_delta=0.20)
        and _is_finger_vertical_extended(hand, 16, 13, min_y_delta=0.05, max_x_delta=0.20)
    )

    thumb_far = _dist2d(thumb_tip, wrist) > 0.24
    pinky_up = _is_finger_vertical_extended(hand, 20, 17, min_y_delta=0.10, max_x_delta=0.13)
    y_like = thumb_far and pinky_up

    compact_o_like = (
        _dist2d(index_tip, middle_tip) < 0.09
        and _dist2d(middle_tip, ring_tip) < 0.09
        and thumb_index_gap < 0.09
    )

    return (
        index_curve
        and middle_curve
        and ring_curve
        and pinky_curve
        and c_gap_ok
        and arc_span_ok
        and thumb_not_looping
        and (not y_like)
        and (not g_like)
        and (not h_like)
        and (not compact_o_like)
    )

def _is_strict_m_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]

    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.26, max_tip_mcp=0.24)
    )
    thumb_under_three = thumb_tip[1] > max(index_tip[1], middle_tip[1], ring_tip[1]) - 0.01
    thumb_near_knuckles = (
        _dist2d(thumb_tip, hand[5]) < 0.20
        and _dist2d(thumb_tip, hand[9]) < 0.20
        and _dist2d(thumb_tip, hand[13]) < 0.21
    )
    thumb_crossing_e = _dist2d(thumb_tip, index_tip) < 0.11 and _dist2d(thumb_tip, middle_tip) < 0.12

    return fist_compact and thumb_under_three and thumb_near_knuckles and not thumb_crossing_e


def _is_loose_m_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.26, max_tip_mcp=0.24)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.26, max_tip_mcp=0.24)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.26, max_tip_mcp=0.24)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.28, max_tip_mcp=0.26)
    )
    tip_cluster = (
        _dist2d(index_tip, middle_tip) < 0.10
        and _dist2d(middle_tip, ring_tip) < 0.10
        and _dist2d(ring_tip, pinky_tip) < 0.12
    )
    thumb_not_crossing_e = not (_dist2d(thumb_tip, index_tip) < 0.11 and _dist2d(thumb_tip, middle_tip) < 0.12)
    thumb_low_or_hidden = thumb_tip[1] > max(index_tip[1], middle_tip[1]) - 0.015

    return fist_compact and tip_cluster and thumb_not_crossing_e and thumb_low_or_hidden


def _is_m_tucked_fist_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.29, max_tip_mcp=0.27)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.29, max_tip_mcp=0.27)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.29, max_tip_mcp=0.27)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.31, max_tip_mcp=0.29)
    )
    finger_row_compact = (
        _dist2d(index_tip, middle_tip) < 0.11
        and _dist2d(middle_tip, ring_tip) < 0.11
        and _dist2d(ring_tip, pinky_tip) < 0.13
    )
    thumb_tucked = thumb_tip[1] > max(index_tip[1], middle_tip[1], ring_tip[1]) - 0.03
    thumb_near_knuckles = (
        _dist2d(thumb_tip, hand[5]) < 0.23
        and _dist2d(thumb_tip, hand[9]) < 0.23
        and _dist2d(thumb_tip, hand[13]) < 0.24
    )

    return fist_compact and finger_row_compact and thumb_tucked and thumb_near_knuckles


def _is_strict_e_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    tip_cluster = (
        _dist2d(index_tip, palm_center) < 0.19
        and _dist2d(middle_tip, palm_center) < 0.19
        and _dist2d(ring_tip, palm_center) < 0.19
        and _dist2d(pinky_tip, palm_center) < 0.19
    )
    folded_all = (
        _is_finger_folded_margin(hand, 8, 6, margin=0.000)
        and _is_finger_folded_margin(hand, 12, 10, margin=0.000)
        and _is_finger_folded_margin(hand, 16, 14, margin=0.000)
        and _is_finger_folded_margin(hand, 20, 18, margin=0.000)
    )
    thumb_crossing = _dist2d(thumb_tip, index_tip) < 0.12 and _dist2d(thumb_tip, middle_tip) < 0.13
    thumb_not_tucked_under = thumb_tip[1] <= ring_tip[1] + 0.010
    return tip_cluster and folded_all and thumb_crossing and thumb_not_tucked_under


def _is_loose_e_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    tip_cluster = (
        _dist2d(index_tip, palm_center) < 0.22
        and _dist2d(middle_tip, palm_center) < 0.22
        and _dist2d(ring_tip, palm_center) < 0.22
        and _dist2d(pinky_tip, palm_center) < 0.22
    )
    folded_all = (
        _is_finger_folded_margin(hand, 8, 6, margin=-0.002)
        and _is_finger_folded_margin(hand, 12, 10, margin=-0.002)
        and _is_finger_folded_margin(hand, 16, 14, margin=-0.002)
        and _is_finger_folded_margin(hand, 20, 18, margin=-0.002)
    )
    thumb_crossing = _dist2d(thumb_tip, index_tip) < 0.14 and _dist2d(thumb_tip, middle_tip) < 0.15
    thumb_not_under = thumb_tip[1] <= ring_tip[1] + 0.03
    return tip_cluster and folded_all and thumb_crossing and thumb_not_under


def _is_e_priority_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    fist_compact = (
        _dist2d(index_tip, palm_center) < 0.22
        and _dist2d(middle_tip, palm_center) < 0.22
        and _dist2d(ring_tip, palm_center) < 0.22
        and _dist2d(pinky_tip, palm_center) < 0.23
    )
    row_compact = (
        _dist2d(index_tip, middle_tip) < 0.12
        and _dist2d(middle_tip, ring_tip) < 0.12
        and _dist2d(ring_tip, pinky_tip) < 0.14
    )
    thumb_crossing = _dist2d(thumb_tip, index_tip) < 0.15 and _dist2d(thumb_tip, middle_tip) < 0.16
    thumb_front = thumb_tip[1] <= ring_tip[1] + 0.04

    return fist_compact and row_compact and thumb_crossing and thumb_front


def _is_strict_n_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]

    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.22, max_tip_mcp=0.20)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
    )
    thumb_under_two = thumb_tip[1] > max(index_tip[1], middle_tip[1]) + 0.004
    thumb_not_under_ring = thumb_tip[1] <= ring_tip[1] + 0.025
    thumb_near_first_two = _dist2d(thumb_tip, hand[5]) < 0.18 and _dist2d(thumb_tip, hand[9]) < 0.18
    thumb_far_from_ring_base = _dist2d(thumb_tip, hand[13]) > 0.085
    thumb_crossing_e = _dist2d(thumb_tip, index_tip) < 0.11 and _dist2d(thumb_tip, middle_tip) < 0.12

    return (
        fist_compact
        and thumb_under_two
        and thumb_not_under_ring
        and thumb_near_first_two
        and thumb_far_from_ring_base
        and not thumb_crossing_e
    )



def _is_loose_n_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False
    hand = hands[0]
    if len(hand) < 21:
        return False

    palm_center = _palm_center(hand)
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]

    fist_compact = (
        _is_finger_compact_to_palm(hand, 8, 5, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 12, 9, palm_center, max_tip_palm=0.24, max_tip_mcp=0.22)
        and _is_finger_compact_to_palm(hand, 16, 13, palm_center, max_tip_palm=0.25, max_tip_mcp=0.23)
        and _is_finger_compact_to_palm(hand, 20, 17, palm_center, max_tip_palm=0.27, max_tip_mcp=0.25)
    )
    thumb_under_two = thumb_tip[1] > max(index_tip[1], middle_tip[1]) - 0.010
    thumb_not_under_ring = thumb_tip[1] <= ring_tip[1] + 0.040
    thumb_near_first_two = _dist2d(thumb_tip, hand[5]) < 0.22 and _dist2d(thumb_tip, hand[9]) < 0.22
    thumb_far_from_ring_base = _dist2d(thumb_tip, hand[13]) > 0.08
    thumb_crossing_e = _dist2d(thumb_tip, index_tip) < 0.105 and _dist2d(thumb_tip, middle_tip) < 0.115
    finger_row_compact = (
        _dist2d(index_tip, middle_tip) < 0.11
        and _dist2d(middle_tip, ring_tip) < 0.11
        and _dist2d(ring_tip, pinky_tip) < 0.13
    )

    return (
        fist_compact
        and finger_row_compact
        and thumb_under_two
        and thumb_not_under_ring
        and thumb_near_first_two
        and thumb_far_from_ring_base
        and not thumb_crossing_e
    )

def _is_checkmark_gesture(landmarks: Dict[str, List]) -> bool:
    hands = landmarks.get("hands", [])
    if len(hands) != 1:
        return False

    hand = hands[0]
    if len(hand) < 21:
        return False

    index_extended = _is_finger_extended(hand, 8, 6)
    middle_folded = not _is_finger_extended(hand, 12, 10)
    ring_folded = not _is_finger_extended(hand, 16, 14)
    pinky_folded = not _is_finger_extended(hand, 20, 18)

    wrist = hand[0]
    index_tip = hand[8]
    thumb_tip = hand[4]

    diagonal_motion = abs(index_tip[0] - wrist[0]) > 0.05 and abs(index_tip[1] - wrist[1]) > 0.05
    thumb_visible = abs(thumb_tip[0] - wrist[0]) > 0.03

    return (
        index_extended
        and middle_folded
        and ring_folded
        and pinky_folded
        and diagonal_motion
        and thumb_visible
    )


def _flatten_dynamic_features(landmarks: Dict[str, List]) -> List[float]:
    frame_features, _ = flatten_dynamic_frame(landmarks)
    return frame_features


def _extract_static_roi(frame, landmarks: Optional[Dict[str, List]]):
    if not landmarks:
        return frame

    hands = landmarks.get("hands", [])
    if not hands:
        return frame

    hand = hands[0]
    if not hand:
        return frame

    frame_h, frame_w = frame.shape[:2]
    xs = [point[0] * frame_w for point in hand]
    ys = [point[1] * frame_h for point in hand]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    hand_size = max(max_x - min_x, max_y - min_y)
    if hand_size <= 2:
        return frame

    padding = hand_size * 0.25

    x1 = max(0, int(min_x - padding))
    y1 = max(0, int(min_y - padding))
    x2 = min(frame_w, int(max_x + padding))
    y2 = min(frame_h, int(max_y + padding))

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    return roi if roi.size else frame


def _compute_motion_score(session: SessionState, landmarks: Dict[str, List]) -> float:
    hands = landmarks.get("hands", [])
    if not hands:
        session.prev_hand_center = None
        session.motion_ema *= 0.9
        return session.motion_ema

    hand = hands[0]
    if not hand:
        session.prev_hand_center = None
        session.motion_ema *= 0.9
        return session.motion_ema

    cx = sum(point[0] for point in hand) / len(hand)
    cy = sum(point[1] for point in hand) / len(hand)

    if session.prev_hand_center is None:
        raw_motion = 0.0
    else:
        px, py = session.prev_hand_center
        dx = cx - px
        dy = cy - py
        raw_motion = (dx * dx + dy * dy) ** 0.5

    session.prev_hand_center = (cx, cy)
    session.motion_ema = 0.7 * session.motion_ema + 0.3 * raw_motion
    return session.motion_ema


def _extract_hand_bbox(landmarks: Dict[str, List]) -> Optional[List[float]]:
    hands = landmarks.get("hands", [])
    if not hands:
        return None

    hand = hands[0]
    if not hand:
        return None

    xs = [point[0] for point in hand]
    ys = [point[1] for point in hand]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    if span <= 0.0:
        return None

    pad = span * 0.2
    x1 = max(0.0, min_x - pad)
    y1 = max(0.0, min_y - pad)
    x2 = min(1.0, max_x + pad)
    y2 = min(1.0, max_y + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    return [float(x1), float(y1), float(x2), float(y2)]


def _enhance_static_roi(roi):
    if roi is None or roi.size == 0:
        return roi

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness >= 95.0:
        return roi

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(cv2.merge((l_eq, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.18, beta=12)
    return enhanced


def _prepare_static_tensor(roi: np.ndarray) -> Optional[torch.Tensor]:
    if roi is None or roi.size == 0:
        return None

    # Slight denoising before equalization.
    roi_blur = cv2.GaussianBlur(roi, (3, 3), sigmaX=0.6)

    # Contrast improvement via histogram equalization on luminance channel.
    ycrcb = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    roi_eq = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)

    # Convert BGR -> RGB and resize to model input size.
    roi_rgb = cv2.cvtColor(roi_eq, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.resize(roi_rgb, (128, 128), interpolation=cv2.INTER_AREA)

    # Normalize pixel range to [0, 1] and convert to tensor (N, C, H, W).
    roi_norm = roi_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(roi_norm, (2, 0, 1))).unsqueeze(0)
    return tensor


def _majority_vote_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None
    counts = Counter(labels)
    label, _votes = counts.most_common(1)[0]
    return label


def _majority_vote_dynamic(labels: List[str], min_votes: Optional[int] = None) -> Optional[str]:
    if not labels:
        return None
    counts = Counter(labels)
    label, votes = counts.most_common(1)[0]
    threshold = max(1, DYNAMIC_VOTE_MIN if min_votes is None else int(min_votes))
    if votes >= threshold:
        return label
    return labels[-1]


def _predict_static(frame, landmarks: Optional[Dict[str, List]] = None) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if STATIC_MODEL is None or not STATIC_CLASSES:
        return None, None, None

    if landmarks is not None and not landmarks.get("hands"):
        return None, None, None

    feature_vector = build_static_feature_vector(landmarks or {})
    if feature_vector is None:
        return None, None, None

    input_tensor = torch.tensor([feature_vector], dtype=torch.float32)

    with torch.inference_mode():
        output = STATIC_MODEL(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf = float(torch.max(probs).item())
        idx = int(torch.argmax(probs, dim=1).item())
        topk = min(2, int(probs.shape[1]))
        top_vals, _ = torch.topk(probs, k=topk, dim=1)
        margin = float(top_vals[0, 0].item() - top_vals[0, 1].item()) if topk > 1 else conf

    return STATIC_CLASSES[idx], conf, margin


def _predict_dynamic_from_sequence(sequence: List[List[float]], conf_threshold: Optional[float] = None) -> Tuple[Optional[str], Optional[float]]:
    if DYNAMIC_MODEL is None:
        return None, None

    if len(sequence) != SEQUENCE_LENGTH:
        return None, None

    input_tensor = torch.tensor([sequence], dtype=torch.float32)

    with torch.inference_mode():
        output = DYNAMIC_MODEL(input_tensor)
        probs = torch.softmax(output, dim=1)
        probs_row = probs[0]
        confidence = float(torch.max(probs_row).item())
        prediction_idx = int(torch.argmax(probs_row).item())
        prediction_label = DYNAMIC_REVERSE_MAP.get(prediction_idx)

    effective_threshold = DYNAMIC_CONFIDENCE_THRESHOLD if conf_threshold is None else max(0.0, min(1.0, conf_threshold))
    if confidence < effective_threshold:
        # Targeted fallback for selected words, with per-label confidence floors.
        if prediction_label in DYNAMIC_FORCE_LABELS and confidence >= _dynamic_force_threshold(prediction_label):
            return prediction_label, confidence

        best_forced_label: Optional[str] = None
        best_forced_conf = 0.0
        for idx in range(int(probs_row.numel())):
            label = DYNAMIC_REVERSE_MAP.get(idx)
            if label not in DYNAMIC_FORCE_LABELS:
                continue
            conf = float(probs_row[idx].item())
            if conf > best_forced_conf:
                best_forced_conf = conf
                best_forced_label = label

        if (
            best_forced_label
            and best_forced_conf >= _dynamic_force_threshold(best_forced_label)
            and (confidence - best_forced_conf) <= DYNAMIC_FORCE_TOP1_MARGIN
        ):
            return best_forced_label, best_forced_conf

        return None, confidence

    return prediction_label, confidence


def process_frame(
    frame,
    session_id: str,
    activation_hold_frames: int,
    no_hand_cooldown_frames: int,
    cooldown_seconds: float,
    require_checkmark_activation: bool = True,
    recognition_mode: str = "hybrid",
) -> Dict[str, object]:
    now = time.monotonic()
    with _session_lock:
        pre_session_status = _session_state[session_id].status
    include_face = pre_session_status == "analyzing"
    landmarks = extract_landmarks(frame, include_face=include_face)

    with _session_lock:
        session = _session_state[session_id]

        hands_detected = bool(landmarks.get("hands"))
        motion_score = _compute_motion_score(session, landmarks)
        if hands_detected:
            session.no_hand_streak = 0
        else:
            session.no_hand_streak += 1

        if session.status == "cooldown" and now >= session.cooldown_until:
            session.status = "standby"
            session.checkmark_streak = 0

        selected_recognition_mode = (recognition_mode or "hybrid").strip().lower()
        if selected_recognition_mode not in {"hybrid", "static", "dynamic"}:
            selected_recognition_mode = "hybrid"

        dynamic_available = (
            DYNAMIC_MODEL is not None and selected_recognition_mode in {"hybrid", "dynamic"}
        )
        static_available = (
            STATIC_MODEL is not None
            and bool(STATIC_CLASSES)
            and selected_recognition_mode in {"hybrid", "static"}
        )

        if session.status == "standby" and dynamic_available:
            if selected_recognition_mode == "dynamic" and hands_detected:
                session.status = "analyzing"
                session.checkmark_streak = 0
                session.clear_dynamic_state()
            elif require_checkmark_activation:
                if _is_checkmark_gesture(landmarks):
                    session.checkmark_streak += 1
                    if session.checkmark_streak >= activation_hold_frames:
                        session.status = "analyzing"
                        session.checkmark_streak = 0
                        session.clear_dynamic_state()
                else:
                    session.checkmark_streak = 0
            elif hands_detected and motion_score >= DYNAMIC_MOTION_THRESHOLD:
                session.status = "analyzing"
                session.checkmark_streak = 0
                session.clear_dynamic_state()

        prediction: Optional[str] = None
        confidence: Optional[float] = None
        speak_text: str = ""
        errors: List[str] = []

        if not dynamic_available and not static_available:
            errors.append("No static or dynamic model is currently loaded.")

        # Stage 1: static prediction on hand-visible frames.
        static_label: Optional[str] = None
        static_conf: Optional[float] = None
        static_margin: Optional[float] = None
        if static_available and hands_detected:
            raw_label, static_conf, static_margin = _predict_static(frame, landmarks)
            static_label = raw_label
            passes_strict_gate = (
                static_conf is not None
                and static_margin is not None
                and static_conf >= STATIC_CONFIDENCE_THRESHOLD
                and static_margin >= STATIC_MIN_MARGIN
            )
            if not passes_strict_gate:
                static_label = None

            model_confident = (
                static_label is not None
                and static_conf is not None
                and static_margin is not None
                and static_conf >= 0.90
                and static_margin >= 0.18
            )

            rule_label = _estimate_static_rule_label(landmarks)
            if rule_label in {"A", "B", "C", "D", "L", "E", "F", "G", "H", "I", "K", "M", "N", "O", "S", "T", "P", "Q", "R", "U", "V", "W", "X", "Y"}:
                confusion_neighborhood = {
                    "A": {"A", "B", "D", "L", "K", None},
                    "B": {"A", "B", "D", "L", "K", None},
                    "C": {"C", "O", "A", "S", "Y", None},
                    "D": {"A", "B", "D", "L", "K", None},
                    "L": {"A", "B", "D", "L", "K", None},
                    "E": {"E", "C", "S", "O", None},
                    "F": {"F", "B", "D", "O", "K", None},
                    "G": {"G", "H", "L", "D", "B", "A", None},
                    "H": {"H", "G", "L", "D", "B", "A", "M", None},
                    "I": {"I", "A", "L", "D", "Y", None},
                    "K": {"K", "A", "D", "L", "B", None},
                    "M": {"M", "E", "A", "N", "S", None},
                    "N": {"N", "E", "M", "A", "S", None},
                    "O": {"O", "E", "C", "A", "S", None},
                    "S": {"S", "A", "E", "M", "N", "T", None},
                    "T": {"T", "S", "A", "M", "N", None},
                    "P": {"P", "K", "D", "Q", "R", "A", None},
                    "Q": {"Q", "G", "P", "D", "A", None},
                    "R": {"R", "K", "U", "V", "A", None},
                    "U": {"U", "V", "R", "K", "A", None},
                    "V": {"V", "U", "R", "K", "A", None},
                    "W": {"W", "V", "U", "R", "A", None},
                    "X": {"X", "A", "S", "T", "D", None},
                    "Y": {"Y", "I", "L", "A", "V", None},
                }
                if (not model_confident) and static_label in confusion_neighborhood[rule_label]:
                    if static_conf is None or static_conf < 0.95 or static_label != rule_label:
                        static_label = rule_label
                        static_conf = max(static_conf or 0.0, 0.82 if rule_label in {"E", "F", "M", "N", "O", "S", "T", "P", "Q", "R"} else 0.80)

            # Hard guard: accept L only when geometry is truly L-shaped.
            if static_label == "L" and not _is_strict_l_gesture(landmarks):
                if rule_label and rule_label != "L":
                    static_label = rule_label
                    static_conf = max(static_conf or 0.0, 0.80 if rule_label != "E" else 0.82)
                else:
                    static_label = None
                    if static_conf is not None:
                        static_conf = min(static_conf, 0.55)

            # Hard guard: if geometry is strict F, do not allow B/D override from model.
            if _is_strict_f_gesture(landmarks):
                if static_label in {None, "B", "D", "O", "K"}:
                    static_label = "F"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: strict H should not become G/L/B/D.
            if _is_strict_h_gesture(landmarks):
                if static_label in {None, "A", "G", "L", "B", "D", "K", "M", "I", "Y"}:
                    static_label = "H"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Soft guard: relaxed H geometry can still recover H from common confusions.
            if _is_loose_h_gesture(landmarks):
                if static_label in {None, "A", "G", "L", "B", "D", "K", "M", "I", "Y"}:
                    if static_conf is None or static_conf < 0.92:
                        static_label = "H"
                        static_conf = max(static_conf or 0.0, 0.82)

            # Hard guard: strict G should not become H/L/B/D.
            if _is_strict_g_gesture(landmarks):
                if static_label in {None, "A", "H", "L", "B", "D", "K", "I", "Y"}:
                    static_label = "G"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Additional D->G correction for sideways index poses.
            if _is_loose_g_gesture(landmarks):
                if static_label in {None, "D", "A", "L", "B", "K", "I", "Y", "H"}:
                    static_label = "G"
                    static_conf = max(static_conf or 0.0, 0.80)

            k_plausible = _is_strict_k_gesture(landmarks) or _is_loose_k_gesture(landmarks)

            # Hard guard: K should not become A/D/L/B/U/V/W.
            if k_plausible:
                if static_label in {None, "A", "D", "L", "B", "U", "V", "W", "R"}:
                    static_label = "K"
                    static_conf = max(static_conf or 0.0, 0.86)

            r_plausible = _is_strict_r_gesture(landmarks) or _is_loose_r_gesture(landmarks)
            p_plausible = _is_strict_p_gesture(landmarks) or _is_loose_p_gesture(landmarks)
            q_plausible = _is_strict_q_gesture(landmarks) or _is_loose_q_gesture(landmarks)

            # Hard guard: R should not collapse to K/A.
            if r_plausible:
                if static_label in {None, "K", "A", "U", "V", "P", "Q"}:
                    static_label = "R"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: P should not collapse to K/D/A.
            if p_plausible and (not _is_strict_a_gesture(landmarks)):
                if static_label in {None, "K", "D", "A", "Q", "R"}:
                    static_label = "P"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: Q should not collapse to G/A/D.
            if q_plausible and (not _is_strict_a_gesture(landmarks)):
                if static_label in {None, "G", "A", "D", "P", "R"}:
                    static_label = "Q"
                    static_conf = max(static_conf or 0.0, 0.84)

            e_plausible = _is_strict_e_gesture(landmarks) or _is_loose_e_gesture(landmarks)
            e_priority = _is_e_priority_gesture(landmarks)

            a_plausible = _is_strict_a_gesture(landmarks)

            # Hard guard: strict A should not collapse to C/P/Q/E/S.
            if a_plausible and (not p_plausible) and (not q_plausible) and (not e_plausible) and (not e_priority):
                if static_label in {None, "C", "P", "Q", "E", "S", "M", "N"}:
                    static_label = "A"
                    static_conf = max(static_conf or 0.0, 0.86)

            d_plausible = _is_strict_d_gesture(landmarks)

            # Hard guard: strict D should not collapse to W/U/V/P/Q/R.
            if d_plausible and (not p_plausible) and (not q_plausible):
                if static_label in {None, "W", "U", "V", "P", "Q", "R", "A", "K"}:
                    static_label = "D"
                    static_conf = max(static_conf or 0.0, 0.88)

            n_plausible = _is_strict_n_gesture(landmarks) or _is_loose_n_gesture(landmarks)

            # Hard guard: clear E should not collapse to T/S/M/N/A/Y.
            if (e_plausible or e_priority) and (not a_plausible) and (not n_plausible):
                if static_label in {None, "T", "S", "M", "N", "Y", "P", "Q"}:
                    static_label = "E"
                    static_conf = max(static_conf or 0.0, 0.88)

            u_plausible = _is_strict_u_gesture(landmarks) or _is_loose_u_gesture(landmarks)
            v_plausible = _is_strict_v_gesture(landmarks) or _is_loose_v_gesture(landmarks)
            w_plausible = _is_strict_w_gesture(landmarks) or _is_loose_w_gesture(landmarks)
            x_plausible = _is_strict_x_gesture(landmarks) or _is_loose_x_gesture(landmarks)
            y_plausible = _is_strict_y_gesture(landmarks) or _is_loose_y_gesture(landmarks)
            c_plausible = _is_strict_c_gesture(landmarks) or _is_loose_c_gesture(landmarks)

            # Hard guard: U should not collapse to A/R/V/K.
            if u_plausible and (not k_plausible):
                if static_label in {None, "A", "R", "V", "K", "H"}:
                    static_label = "U"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: V should not collapse to A/U/R/K.
            if v_plausible and (not k_plausible):
                if static_label in {None, "A", "U", "R", "K", "H"}:
                    static_label = "V"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: W should not collapse to V/U/R/A.
            if w_plausible and (not k_plausible):
                if static_label in {None, "V", "U", "R", "A", "H"}:
                    static_label = "W"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: clear K should not collapse to V/W/U.
            if k_plausible and static_label in {"V", "W", "U", "R"}:
                static_label = "K"
                static_conf = max(static_conf or 0.0, 0.86)

            # Hard guard: X should not collapse to A/S/T/D.
            if x_plausible:
                if static_label in {None, "A", "S", "T", "D"}:
                    static_label = "X"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: clear C should not collapse to G/H/Y/O/A/S.
            if _is_strict_c_gesture(landmarks):
                if static_label in {None, "G", "H", "W", "Y", "O", "A", "S", "Q", "N"}:
                    static_label = "C"
                    static_conf = max(static_conf or 0.0, 0.90)
            elif c_plausible and (not y_plausible) and (not _is_strict_g_gesture(landmarks)) and (not _is_strict_h_gesture(landmarks)):
                if static_label in {None, "Y", "O", "A", "S", "Q", "G", "H", "W", "N"}:
                    static_label = "C"
                    static_conf = max(static_conf or 0.0, 0.86)

            # Hard guard: Y should not collapse to I/L/A/V/C.
            if y_plausible and (not a_plausible) and (not e_plausible) and (not c_plausible):
                if static_label in {None, "I", "L", "A", "V", "U", "C", "O"}:
                    static_label = "Y"
                    static_conf = max(static_conf or 0.0, 0.84)

            o_plausible = _is_strict_o_gesture(landmarks) or _is_loose_o_gesture(landmarks)
            s_plausible = _is_strict_s_gesture(landmarks) or _is_loose_s_gesture(landmarks)
            t_plausible = _is_strict_t_gesture(landmarks) or _is_loose_t_gesture(landmarks)
            g_plausible = _is_strict_g_gesture(landmarks) or _is_loose_g_gesture(landmarks)
            h_plausible = _is_strict_h_gesture(landmarks) or _is_loose_h_gesture(landmarks)

            # Final C enforcement: if the hand still looks like a rounded C after
            # O/G/H checks, let C win even when the model is uncertain.
            c_force = c_plausible and (not w_plausible) and (not y_plausible) and (not o_plausible) and (not h_plausible) and (not g_plausible)
            if (not model_confident) and c_force and static_label in {None, "G", "H", "W", "Y", "O", "A", "S", "Q", "N", "D", "U", "V"}:
                static_label = "C"
                static_conf = max(static_conf or 0.0, 0.92)

            # Hard guard: O should not collapse to E/A/C/S.
            if o_plausible and (not c_plausible):
                if static_label in {None, "E", "A", "C", "S"}:
                    static_label = "O"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: S should not collapse to A/E/M/N/T.
            if s_plausible and (not e_plausible) and (not e_priority):
                if static_label in {None, "A", "E", "M", "N", "T"}:
                    static_label = "S"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: T should not collapse to A/S/M/N.
            if t_plausible and (not a_plausible) and (not e_plausible) and (not e_priority) and (not g_plausible) and (not h_plausible):
                if static_label in {None, "A", "S", "M", "N", "E"}:
                    static_label = "T"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: clear H should not collapse to T.
            if h_plausible:
                if static_label in {None, "T", "A", "G", "L", "B", "D", "K", "M", "I", "Y"}:
                    static_label = "H"
                    static_conf = max(static_conf or 0.0, 0.86)

            # Hard guard: clear G should not collapse to T.
            if g_plausible and (not c_plausible):
                if static_label in {None, "T", "A", "D", "K", "S", "I", "Y", "H"}:
                    static_label = "G"
                    static_conf = max(static_conf or 0.0, 0.86)

            # Hard guard: strict M should not become E/A/N/S.
            if (_is_strict_m_gesture(landmarks) or _is_m_tucked_fist_gesture(landmarks)) and (not n_plausible) and (not e_plausible) and (not e_priority):
                if static_label in {None, "E", "A", "N", "S"}:
                    static_label = "M"
                    static_conf = max(static_conf or 0.0, 0.84)

            # E vs M disambiguation: if model says E but E geometry is weak and
            # M geometry is plausible, force M.
            if static_label == "E" and (not e_plausible):
                m_plausible = (
                    _is_strict_m_gesture(landmarks)
                    or _is_loose_m_gesture(landmarks)
                    or _is_m_tucked_fist_gesture(landmarks)
                )
                weak_e_signal = (
                    static_conf is None
                    or static_conf < 0.90
                    or (static_margin is not None and static_margin < 0.11)
                )
                if m_plausible and (not n_plausible) and (
                    (not _is_strict_e_gesture(landmarks))
                    or weak_e_signal
                    or _is_m_tucked_fist_gesture(landmarks)
                ):
                    static_label = "M"
                    static_conf = max(static_conf or 0.0, 0.84)

            # Hard guard: N should not become E/M/A/S.
            if n_plausible:
                if static_label in {None, "E", "M", "A", "S"}:
                    static_label = "N"
                    static_conf = max(static_conf or 0.0, 0.88)

            # Hard guard: prevent one-finger gestures collapsing to A.
            if static_label == "A":
                if k_plausible:
                    static_label = "K"
                    static_conf = max(static_conf or 0.0, 0.82)
                elif _is_strict_i_gesture(landmarks):
                    static_label = "I"
                    static_conf = max(static_conf or 0.0, 0.82)
                else:
                    rule_now = _estimate_static_rule_label(landmarks)
                    if rule_now in {"D", "G", "H", "L", "K", "M", "N", "O", "S", "T", "P", "Q", "R", "U", "V", "W", "X", "Y"}:
                        static_label = rule_now
                        static_conf = max(static_conf or 0.0, 0.80)

            # If geometry strongly indicates G/H but model gate is weak, prefer
            # rule label over blank output.
            if static_label is None and rule_label in {"C", "G", "H", "Q", "O", "U", "V", "W", "Y"}:
                if static_conf is None or static_conf >= 0.40:
                    static_label = rule_label
                    static_conf = max(static_conf or 0.0, 0.76)

            if static_label is None and rule_label in {"M", "N", "P", "R", "S", "T", "X"}:
                if static_conf is None or static_conf >= 0.38:
                    static_label = rule_label
                    static_conf = max(static_conf or 0.0, 0.76)

            # Soft fallback to prevent blank output when hand is detected but
            # confidence is slightly below strict threshold.
            if (
                static_label is None
                and raw_label is not None
                and static_conf is not None
                and static_margin is not None
                and static_conf >= max(0.50, STATIC_CONFIDENCE_THRESHOLD - 0.12)
                and static_margin >= max(0.03, STATIC_MIN_MARGIN - 0.04)
                and raw_label not in {"A", "L"}
            ):
                static_label = raw_label

            # Never allow a weak/non-strict L via soft fallback.
            if static_label == "L" and not _is_strict_l_gesture(landmarks):
                static_label = None

            if static_label:
                if session.static_candidate_label == static_label:
                    session.static_candidate_streak += 1
                else:
                    session.static_candidate_label = static_label
                    session.static_candidate_streak = 1

                if session.static_candidate_streak >= STATIC_STABLE_FRAMES:
                    session.static_recent_labels.append(static_label)
                    if len(session.static_recent_labels) > max(1, STATIC_PREDICTION_WINDOW):
                        session.static_recent_labels = session.static_recent_labels[-max(1, STATIC_PREDICTION_WINDOW):]

                    voted_label = _majority_vote_label(session.static_recent_labels)
                    can_update_display = (
                        static_conf is not None
                        and static_conf >= STATIC_DISPLAY_CONFIDENCE_THRESHOLD
                    )

                    if can_update_display and voted_label:
                        session.last_static_label = voted_label
                        prediction = voted_label
                    else:
                        prediction = session.last_static_label or prediction

                    confidence = static_conf if static_conf is not None else confidence
            else:
                session.reset_static_candidate()
                if static_conf is not None:
                    confidence = static_conf
        elif not hands_detected:
            session.reset_static_candidate()

        dynamic_mode_selected = selected_recognition_mode == "dynamic"
        effective_vote_window = 1 if dynamic_mode_selected else max(1, DYNAMIC_VOTE_WINDOW)
        effective_vote_min = 1 if dynamic_mode_selected else max(1, DYNAMIC_VOTE_MIN)
        effective_stable_windows = 1 if dynamic_mode_selected else max(1, DYNAMIC_STABLE_WINDOWS)
        effective_emit_cooldown = 0.35 if dynamic_mode_selected else DYNAMIC_EMIT_COOLDOWN_SECONDS
        effective_dynamic_threshold = 0.30 if dynamic_mode_selected else DYNAMIC_CONFIDENCE_THRESHOLD

        # Stage 2: dynamic prediction only when motion is high.
        if session.status == "analyzing":
            if DYNAMIC_SUPPRESS_WHEN_STATIC and static_label and selected_recognition_mode != "dynamic":
                # Avoid dynamic words (e.g., "no") overriding active static alphabet detection.
                session.clear_dynamic_state()
            elif not dynamic_available:
                session.status = "standby"
                session.clear_dynamic_state()
            elif session.no_hand_streak >= no_hand_cooldown_frames:
                session.status = "cooldown"
                session.cooldown_until = now + cooldown_seconds
                session.clear_dynamic_state()
            else:
                session.sequence.append(_flatten_dynamic_features(landmarks))
                session.sequence_hand_presence.append(1 if hands_detected else 0)
                session.sequence_motion_scores.append(motion_score)
                if len(session.sequence) > SEQUENCE_LENGTH:
                    session.sequence = session.sequence[-SEQUENCE_LENGTH:]
                    session.sequence_hand_presence = session.sequence_hand_presence[-SEQUENCE_LENGTH:]
                    session.sequence_motion_scores = session.sequence_motion_scores[-SEQUENCE_LENGTH:]

                if len(session.sequence) == SEQUENCE_LENGTH:
                    avg_motion = sum(session.sequence_motion_scores) / max(
                        len(session.sequence_motion_scores), 1
                    )
                    dynamic_motion_ok = (
                        selected_recognition_mode == "dynamic"
                        or avg_motion >= DYNAMIC_MOTION_THRESHOLD
                    )
                    if (
                        sum(session.sequence_hand_presence) < DYNAMIC_MIN_HAND_FRAMES
                        or not dynamic_motion_ok
                    ):
                        dynamic_label, dynamic_conf = None, None
                    else:
                        dynamic_label, dynamic_conf = _predict_dynamic_from_sequence(session.sequence, conf_threshold=effective_dynamic_threshold)

                    stride = max(1, DYNAMIC_WINDOW_STRIDE)
                    session.sequence = session.sequence[stride:]
                    session.sequence_hand_presence = session.sequence_hand_presence[stride:]
                    session.sequence_motion_scores = session.sequence_motion_scores[stride:]

                    if dynamic_label:
                        session.dynamic_recent_labels.append(dynamic_label)
                        if len(session.dynamic_recent_labels) > effective_vote_window:
                            session.dynamic_recent_labels = session.dynamic_recent_labels[-effective_vote_window:]

                        voted_dynamic = _majority_vote_dynamic(session.dynamic_recent_labels, min_votes=effective_vote_min)
                        if voted_dynamic == session.dynamic_candidate_label:
                            session.dynamic_candidate_streak += 1
                        else:
                            session.dynamic_candidate_label = voted_dynamic or ""
                            session.dynamic_candidate_streak = 1

                        required_stable_windows = effective_stable_windows
                        if voted_dynamic in DYNAMIC_FORCE_LABELS:
                            required_stable_windows = max(effective_stable_windows, DYNAMIC_FORCE_STABLE_WINDOWS)

                        force_conf_ok = True
                        if voted_dynamic in DYNAMIC_FORCE_LABELS:
                            force_conf_ok = (dynamic_conf or 0.0) >= _dynamic_force_threshold(voted_dynamic)

                        can_emit = (
                            bool(voted_dynamic)
                            and force_conf_ok
                            and session.dynamic_candidate_streak >= required_stable_windows
                            and (now - session.last_dynamic_emit_ts) >= effective_emit_cooldown
                        )
                        if can_emit:
                            if not session.words or session.words[-1] != voted_dynamic:
                                session.words.append(voted_dynamic)
                            session.last_dynamic_emit_ts = now
                            session.no_dynamic_prediction_windows = 0
                            prediction = voted_dynamic
                            confidence = dynamic_conf
                            speak_text = correct_sentence(session.words)
                        else:
                            session.no_dynamic_prediction_windows += 1
                    else:
                        session.dynamic_candidate_streak = 0
                        session.no_dynamic_prediction_windows += 1

                    if session.no_dynamic_prediction_windows >= DYNAMIC_MAX_NO_PRED_WINDOWS:
                        session.status = "standby"
                        session.clear_dynamic_state()

                    if dynamic_conf is not None:
                        confidence = dynamic_conf

        if session.status != "analyzing":
            if static_label:
                if session.static_candidate_streak >= STATIC_STABLE_FRAMES:
                    prediction = static_label
                    confidence = static_conf if static_conf is not None else confidence
            elif static_conf is not None:
                confidence = static_conf
            elif not hands_detected:
                prediction = session.last_static_label or prediction
            else:
                prediction = prediction if prediction is not None else None

        sentence = correct_sentence(session.words)
        hand_bbox = _extract_hand_bbox(landmarks)

        return {
            "status": session.status,
            "prediction": prediction,
            "confidence": confidence,
            "sentence": sentence,
            "speak_text": speak_text,
            "hand_bbox": hand_bbox,
            "hand_detected": hands_detected,
            "errors": errors,
        }


def process_video(video_path: str) -> Dict[str, object]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {
            "status": "failed",
            "prediction": None,
            "confidence": None,
            "sentence": "",
            "frames_processed": 0,
            "errors": ["Unable to open uploaded video file."],
        }

    frame_count = 0
    sequence: List[List[float]] = []
    sequence_hand_presence: List[int] = []
    dynamic_recent_labels: List[str] = []
    dynamic_candidate_label = ""
    dynamic_candidate_streak = 0
    words: List[str] = []
    last_confidence: Optional[float] = None
    last_prediction: Optional[str] = None

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_count += 1
        if frame_count % max(VIDEO_FRAME_STRIDE, 1) != 0:
            continue

        landmarks = extract_landmarks(frame, include_face=True)
        if not landmarks.get("hands"):
            continue

        sequence.append(_flatten_dynamic_features(landmarks))
        sequence_hand_presence.append(1 if landmarks.get("hands") else 0)
        if len(sequence) > SEQUENCE_LENGTH:
            sequence = sequence[-SEQUENCE_LENGTH:]
            sequence_hand_presence = sequence_hand_presence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            if sum(sequence_hand_presence) < DYNAMIC_MIN_HAND_FRAMES:
                dynamic_label, dynamic_conf = None, None
            else:
                dynamic_label, dynamic_conf = _predict_dynamic_from_sequence(sequence, conf_threshold=DYNAMIC_CONFIDENCE_THRESHOLD)

            stride = max(1, DYNAMIC_WINDOW_STRIDE)
            sequence = sequence[stride:]
            sequence_hand_presence = sequence_hand_presence[stride:]

            if dynamic_conf is not None:
                last_confidence = dynamic_conf
            if dynamic_label:
                dynamic_recent_labels.append(dynamic_label)
                if len(dynamic_recent_labels) > max(1, DYNAMIC_VOTE_WINDOW):
                    dynamic_recent_labels = dynamic_recent_labels[-max(1, DYNAMIC_VOTE_WINDOW):]

                voted_dynamic = _majority_vote_dynamic(dynamic_recent_labels)
                if voted_dynamic == dynamic_candidate_label:
                    dynamic_candidate_streak += 1
                else:
                    dynamic_candidate_label = voted_dynamic or ""
                    dynamic_candidate_streak = 1

                if voted_dynamic and dynamic_candidate_streak >= max(1, DYNAMIC_STABLE_WINDOWS):
                    if not words or words[-1] != voted_dynamic:
                        words.append(voted_dynamic)
                    last_prediction = voted_dynamic

    capture.release()

    sentence = correct_sentence(words)
    status = "completed" if words else "no_prediction"

    return {
        "status": status,
        "prediction": last_prediction,
        "confidence": last_confidence,
        "sentence": sentence,
        "hand_bbox": None,
        "hand_detected": None,
        "frames_processed": frame_count,
        "errors": [],
    }























































