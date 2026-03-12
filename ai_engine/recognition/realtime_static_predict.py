import json
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from ai_engine.preprocessing.landmark_extractor import extract_landmarks


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "static_gesture_model.h5"
CLASS_MAP_PATH = ROOT_DIR / "models" / "static_class_map.json"
DATASET_DIR = ROOT_DIR / "datasets" / "static" / "asl_alphabet"

IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.75
SMOOTH_WINDOW = 10
BLURRY_LAPLACIAN_THRESHOLD = 60.0
MARGIN_THRESHOLD = 0.15
CONFIRM_FRAMES = 15


def _load_class_names() -> List[str]:
    if CLASS_MAP_PATH.exists():
        return json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Class map missing ({CLASS_MAP_PATH}) and dataset folder not found ({DATASET_DIR})."
        )

    return sorted([p.name for p in DATASET_DIR.iterdir() if p.is_dir()])


def _bbox_from_landmarks(
    hand_landmarks: List[List[float]], frame_width: int, frame_height: int, pad_px: int = 20
) -> Optional[Tuple[int, int, int, int]]:
    if not hand_landmarks:
        return None

    xs = [int(lm[0] * frame_width) for lm in hand_landmarks]
    ys = [int(lm[1] * frame_height) for lm in hand_landmarks]

    x1 = max(0, min(xs) - pad_px)
    y1 = max(0, min(ys) - pad_px)
    x2 = min(frame_width - 1, max(xs) + pad_px)
    y2 = min(frame_height - 1, max(ys) + pad_px)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _expand_bbox(
    bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int, expand_ratio: float = 0.20
) -> Tuple[int, int, int, int]:
    """Expand bbox by 20% for cropping only (drawing bbox remains unchanged)."""
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    dx = int(bw * expand_ratio)
    dy = int(bh * expand_ratio)

    ex1 = max(0, x1 - dx)
    ey1 = max(0, y1 - dy)
    ex2 = min(frame_width - 1, x2 + dx)
    ey2 = min(frame_height - 1, y2 + dy)

    return ex1, ey1, ex2, ey2


def _apply_skin_mask_hsv(roi_bgr: np.ndarray) -> np.ndarray:
    """Suppress background using HSV skin color masking."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    lower_skin_1 = np.array([0, 30, 40], dtype=np.uint8)
    upper_skin_1 = np.array([25, 200, 255], dtype=np.uint8)

    lower_skin_2 = np.array([160, 30, 40], dtype=np.uint8)
    upper_skin_2 = np.array([179, 200, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_skin_1, upper_skin_1)
    mask2 = cv2.inRange(hsv, lower_skin_2, upper_skin_2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask)


def _preprocess_roi(roi_bgr: np.ndarray) -> np.ndarray:
    masked_roi = _apply_skin_mask_hsv(roi_bgr)
    roi_resized = cv2.resize(masked_roi, IMG_SIZE, interpolation=cv2.INTER_AREA)

    roi_blurred = cv2.GaussianBlur(roi_resized, (3, 3), 0)

    ycrcb = cv2.cvtColor(roi_blurred, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y_channel)
    roi_eq_bgr = cv2.cvtColor(cv2.merge((y_eq, cr_channel, cb_channel)), cv2.COLOR_YCrCb2BGR)

    roi_rgb = cv2.cvtColor(roi_eq_bgr, cv2.COLOR_BGR2RGB)
    roi_norm = roi_rgb.astype(np.float32) / 255.0
    return np.expand_dims(roi_norm, axis=0)


def _is_blurry(roi_bgr: np.ndarray, threshold: float = BLURRY_LAPLACIAN_THRESHOLD) -> bool:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness < threshold


def _dist2d(p1: List[float], p2: List[float]) -> float:
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def _is_extended(hand: List[List[float]], tip: int, pip: int, margin: float = 0.012) -> bool:
    return hand[tip][1] < (hand[pip][1] - margin)


def _is_folded(hand: List[List[float]], tip: int, pip: int, margin: float = 0.008) -> bool:
    return hand[tip][1] > (hand[pip][1] + margin)


def _finger_states(hand: List[List[float]]) -> Dict[str, object]:
    wrist = hand[0]
    thumb_tip = hand[4]

    index_tip, index_pip, index_mcp = hand[8], hand[6], hand[5]
    middle_tip, middle_pip, middle_mcp = hand[12], hand[10], hand[9]
    ring_tip, ring_pip, ring_mcp = hand[16], hand[14], hand[13]
    pinky_tip, pinky_pip, pinky_mcp = hand[20], hand[18], hand[17]

    index_up = _is_extended(hand, 8, 6)
    middle_up = _is_extended(hand, 12, 10)
    ring_up = _is_extended(hand, 16, 14)
    pinky_up = _is_extended(hand, 20, 18)

    index_folded = _is_folded(hand, 8, 6)
    middle_folded = _is_folded(hand, 12, 10)
    ring_folded = _is_folded(hand, 16, 14)
    pinky_folded = _is_folded(hand, 20, 18)

    fist_like = index_folded and middle_folded and ring_folded and pinky_folded

    thumb_near_index = _dist2d(thumb_tip, index_mcp) < 0.16
    thumb_near_middle = _dist2d(thumb_tip, middle_mcp) < 0.16
    thumb_near_ring = _dist2d(thumb_tip, ring_mcp) < 0.17

    thumb_under_two = thumb_tip[1] > max(index_tip[1], middle_tip[1]) + 0.01
    thumb_under_three = thumb_tip[1] > max(index_tip[1], middle_tip[1], ring_tip[1]) + 0.01
    thumb_not_under_ring = thumb_tip[1] <= ring_tip[1] + 0.015

    thumb_cross_front = _dist2d(thumb_tip, index_tip) < 0.11 and _dist2d(thumb_tip, middle_tip) < 0.13
    thumb_open_side = _dist2d(thumb_tip, index_mcp) > 0.09 and _dist2d(thumb_tip, wrist) > 0.12

    index_middle_gap = abs(index_tip[0] - middle_tip[0])

    one_finger_index = index_up and middle_folded and ring_folded and pinky_folded

    # Curvature cues for C/O-like shapes.
    palm_center = [
        (hand[5][0] + hand[9][0] + hand[13][0] + hand[17][0]) / 4.0,
        (hand[5][1] + hand[9][1] + hand[13][1] + hand[17][1]) / 4.0,
    ]
    index_thumb_tip_dist = _dist2d(index_tip, thumb_tip)
    index_palm = _dist2d(index_tip, palm_center)
    middle_palm = _dist2d(middle_tip, palm_center)
    ring_palm = _dist2d(ring_tip, palm_center)
    pinky_palm = _dist2d(pinky_tip, palm_center)

    # Downward orientation cues for P/Q.
    index_down = (index_tip[1] - hand[5][1]) > 0.06
    middle_down = (middle_tip[1] - hand[9][1]) > 0.06

    # R-crossing cue: tip order differs from PIP order.
    crossed_index_middle = ((index_tip[0] - middle_tip[0]) * (index_pip[0] - middle_pip[0])) < 0.0

    thumb_pinky_far = _dist2d(thumb_tip, pinky_tip) > 0.22
    thumb_wrist_far = _dist2d(thumb_tip, wrist) > 0.16

    index_hooked = index_tip[1] > index_pip[1] and index_tip[1] < hand[5][1] + 0.02

    return {
        "index_up": index_up,
        "middle_up": middle_up,
        "ring_up": ring_up,
        "pinky_up": pinky_up,
        "index_folded": index_folded,
        "middle_folded": middle_folded,
        "ring_folded": ring_folded,
        "pinky_folded": pinky_folded,
        "fist_like": fist_like,
        "thumb_near_index": thumb_near_index,
        "thumb_near_middle": thumb_near_middle,
        "thumb_near_ring": thumb_near_ring,
        "thumb_under_two": thumb_under_two,
        "thumb_under_three": thumb_under_three,
        "thumb_not_under_ring": thumb_not_under_ring,
        "thumb_cross_front": thumb_cross_front,
        "thumb_open_side": thumb_open_side,
        "index_middle_gap": index_middle_gap,
        "one_finger_index": one_finger_index,
        "index_thumb_tip_dist": index_thumb_tip_dist,
        "index_palm": index_palm,
        "middle_palm": middle_palm,
        "ring_palm": ring_palm,
        "pinky_palm": pinky_palm,
        "index_down": index_down,
        "middle_down": middle_down,
        "crossed_index_middle": crossed_index_middle,
        "thumb_pinky_far": thumb_pinky_far,
        "thumb_wrist_far": thumb_wrist_far,
        "index_hooked": index_hooked,
    }


def _rule_c(s: Dict[str, object]) -> bool:
    return bool(
        0.10 <= float(s["index_thumb_tip_dist"]) <= 0.30
        and float(s["index_palm"]) > 0.10
        and float(s["middle_palm"]) > 0.10
        and float(s["ring_palm"]) > 0.10
        and float(s["pinky_palm"]) > 0.10
    )


def _rule_m(s: Dict[str, object]) -> bool:
    return bool(
        s["fist_like"]
        and s["thumb_under_three"]
        and s["thumb_near_index"]
        and s["thumb_near_middle"]
        and s["thumb_near_ring"]
    )


def _rule_n(s: Dict[str, object]) -> bool:
    return bool(
        s["fist_like"]
        and s["thumb_under_two"]
        and s["thumb_not_under_ring"]
        and s["thumb_near_index"]
        and s["thumb_near_middle"]
    )


def _rule_p(s: Dict[str, object]) -> bool:
    return bool(
        s["index_down"]
        and s["middle_down"]
        and s["ring_folded"]
        and s["pinky_folded"]
        and s["thumb_near_index"]
        and s["thumb_near_middle"]
    )


def _rule_q(s: Dict[str, object]) -> bool:
    return bool(
        s["index_down"]
        and s["middle_folded"]
        and s["ring_folded"]
        and s["pinky_folded"]
        and s["thumb_near_index"]
    )


def _rule_r(s: Dict[str, object]) -> bool:
    return bool(
        s["index_up"]
        and s["middle_up"]
        and s["ring_folded"]
        and s["pinky_folded"]
        and s["crossed_index_middle"]
    )


def _rule_s(s: Dict[str, object]) -> bool:
    return bool(s["fist_like"] and s["thumb_cross_front"] and not s["thumb_under_three"])


def _rule_t(s: Dict[str, object]) -> bool:
    return bool(
        s["fist_like"]
        and s["thumb_near_index"]
        and s["thumb_near_middle"]
        and not s["thumb_near_ring"]
    )


def _rule_u(s: Dict[str, object]) -> bool:
    return bool(
        s["index_up"]
        and s["middle_up"]
        and s["ring_folded"]
        and s["pinky_folded"]
        and float(s["index_middle_gap"]) <= 0.060
    )


def _rule_v(s: Dict[str, object]) -> bool:
    return bool(
        s["index_up"]
        and s["middle_up"]
        and s["ring_folded"]
        and s["pinky_folded"]
        and float(s["index_middle_gap"]) >= 0.075
    )


def _rule_w(s: Dict[str, object]) -> bool:
    return bool(s["index_up"] and s["middle_up"] and s["ring_up"] and s["pinky_folded"])


def _rule_x(s: Dict[str, object]) -> bool:
    return bool(s["index_hooked"] and s["middle_folded"] and s["ring_folded"] and s["pinky_folded"])


def _rule_y(s: Dict[str, object]) -> bool:
    return bool(
        s["pinky_up"]
        and s["index_folded"]
        and s["middle_folded"]
        and s["ring_folded"]
        and s["thumb_pinky_far"]
        and s["thumb_wrist_far"]
    )


LETTER_RULES: Dict[str, Callable[[Dict[str, object]], bool]] = {
    "C": _rule_c,
    "M": _rule_m,
    "N": _rule_n,
    "P": _rule_p,
    "Q": _rule_q,
    "R": _rule_r,
    "S": _rule_s,
    "T": _rule_t,
    "U": _rule_u,
    "V": _rule_v,
    "W": _rule_w,
    "X": _rule_x,
    "Y": _rule_y,
}


def _rule_validate_prediction(pred_label: str, hands: List[List[List[float]]]) -> bool:
    rule = LETTER_RULES.get(pred_label.upper())
    if rule is None:
        return True

    if not hands or len(hands[0]) < 21:
        return False

    states = _finger_states(hands[0])
    return bool(rule(states))


def run_realtime_static_demo() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    class_names = _load_class_names()
    model = tf.keras.models.load_model(str(MODEL_PATH))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    prediction_window = deque(maxlen=SMOOTH_WINDOW)
    last_majority_label = None
    majority_streak = 0
    confirmed_label = "Detecting..."
    confirmed_conf = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w = frame.shape[:2]
        landmarks_data = extract_landmarks(frame, include_face=False)
        hands = landmarks_data.get("hands", [])

        accepted_label = None
        accepted_conf = 0.0

        if hands:
            base_bbox = _bbox_from_landmarks(hands[0], frame_w, frame_h, pad_px=20)
            if base_bbox is not None:
                # Keep existing green bbox drawing logic unchanged.
                bx1, by1, bx2, by2 = base_bbox
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 255, 120), 2)

                # Expand bbox by 20% for crop used in model inference.
                x1, y1, x2, y2 = _expand_bbox(base_bbox, frame_w, frame_h, expand_ratio=0.20)

                hand_roi = frame[y1:y2, x1:x2]
                if hand_roi.size > 0 and not _is_blurry(hand_roi):
                    model_input = _preprocess_roi(hand_roi)
                    logits_or_probs = model.predict(model_input, verbose=0)[0]
                    probs = tf.nn.softmax(logits_or_probs).numpy()

                    pred_idx = int(np.argmax(probs))
                    pred_conf = float(probs[pred_idx])
                    sorted_probs = np.sort(probs)
                    second_prob = float(sorted_probs[-2]) if sorted_probs.shape[0] > 1 else 0.0
                    margin = pred_conf - second_prob

                    if pred_conf >= CONFIDENCE_THRESHOLD and margin > MARGIN_THRESHOLD:
                        candidate = class_names[pred_idx]
                        if _rule_validate_prediction(candidate, hands):
                            accepted_label = candidate
                            accepted_conf = pred_conf
        else:
            prediction_window.clear()
            last_majority_label = None
            majority_streak = 0
            confirmed_label = "Detecting..."
            confirmed_conf = 0.0

        if accepted_label is not None:
            prediction_window.append(accepted_label)
            majority_label = Counter(prediction_window).most_common(1)[0][0]

            if majority_label == last_majority_label:
                majority_streak += 1
            else:
                last_majority_label = majority_label
                majority_streak = 1

            if majority_streak >= CONFIRM_FRAMES:
                confirmed_label = majority_label
                confirmed_conf = accepted_conf

        label_text = (
            f"{confirmed_label} ({confirmed_conf:.2f})"
            if confirmed_label != "Detecting..."
            else "Detecting..."
        )

        cv2.putText(
            frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Static Gesture Recognition", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_static_demo()
