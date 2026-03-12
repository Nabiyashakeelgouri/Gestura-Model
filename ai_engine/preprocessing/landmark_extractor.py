import os
from threading import Lock
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, min(1.0, parsed))


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


HAND_MAX_NUM = _env_int("HAND_MAX_NUM", 2)
HAND_DETECTION_CONFIDENCE = _env_float("HAND_DETECTION_CONFIDENCE", 0.45)
HAND_TRACKING_CONFIDENCE = _env_float("HAND_TRACKING_CONFIDENCE", 0.40)
FACE_DETECTION_CONFIDENCE = _env_float("FACE_DETECTION_CONFIDENCE", 0.60)
FACE_TRACKING_CONFIDENCE = _env_float("FACE_TRACKING_CONFIDENCE", 0.50)
MEDIAPIPE_MODEL_COMPLEXITY = _env_int("MEDIAPIPE_MODEL_COMPLEXITY", 1, minimum=0)
HAND_MIN_SPAN = _env_float("HAND_MIN_SPAN", 0.045)
HAND_MIN_SPAN_DARK = _env_float("HAND_MIN_SPAN_DARK", 0.028)
ENABLE_LOW_LIGHT_HAND_FALLBACK = os.getenv("ENABLE_LOW_LIGHT_HAND_FALLBACK", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HAND_CACHE_MAX_MISSES = _env_int("HAND_CACHE_MAX_MISSES", 5)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh


def _create_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=HAND_MAX_NUM,
        model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )


def _create_face_mesh():
    return mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=FACE_DETECTION_CONFIDENCE,
        min_tracking_confidence=FACE_TRACKING_CONFIDENCE,
    )


hands = _create_hands()
face_mesh = _create_face_mesh()
_mediapipe_lock = Lock()
_last_hands_cache: List[List[List[float]]] = []
_last_hands_misses = 0


def _enhance_low_light_for_detection(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_eq, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def _gamma_enhance(frame, gamma: float):
    if gamma <= 0:
        return frame
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)


def _process_hands_and_face(frame_bgr, include_face: bool = True):
    global face_mesh
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    try:
        with _mediapipe_lock:
            hand_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb) if include_face else None
        return hand_results, face_results
    except ValueError as exc:
        if "Packet timestamp mismatch" not in str(exc):
            raise
        with _mediapipe_lock:
            try:
                face_mesh.close()
            except Exception:
                pass
            face_mesh = _create_face_mesh()
            hand_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb) if include_face else None
        return hand_results, face_results
    finally:
        frame_rgb.flags.writeable = True


def extract_landmarks(frame, include_face: bool = True):
    global _last_hands_cache, _last_hands_misses
    hand_results, face_results = _process_hands_and_face(frame, include_face=include_face)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_brightness = float(frame_gray.mean())

    if ENABLE_LOW_LIGHT_HAND_FALLBACK and not hand_results.multi_hand_landmarks:
        candidates = [_enhance_low_light_for_detection(frame), cv2.convertScaleAbs(frame, alpha=1.2, beta=20)]
        if frame_brightness < 80.0:
            candidates.append(_gamma_enhance(frame, gamma=1.8))

        for candidate in candidates:
            enhanced_hands, enhanced_face = _process_hands_and_face(candidate, include_face=include_face)
            if enhanced_hands.multi_hand_landmarks:
                hand_results = enhanced_hands
            if enhanced_face is not None and enhanced_face.multi_face_landmarks:
                face_results = enhanced_face
            if hand_results.multi_hand_landmarks:
                break

    hands_landmarks = []
    face_landmarks = []

    if hand_results.multi_hand_landmarks:
        hand_entries: List[Tuple[int, float, List[List[float]]]] = []
        rejected_entries: List[Tuple[float, int, float, List[List[float]]]] = []
        handedness_entries = list(hand_results.multi_handedness or [])

        for idx, hand in enumerate(hand_results.multi_hand_landmarks):
            hand_points = []
            sum_x = 0.0
            xs = []
            ys = []

            for lm in hand.landmark:
                sum_x += lm.x
                xs.append(lm.x)
                ys.append(lm.y)
                hand_points.append([lm.x, lm.y, lm.z])

            hand_span = max(max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else 0.0
            base_threshold = HAND_MIN_SPAN_DARK if frame_brightness < 80.0 else HAND_MIN_SPAN
            span_threshold = base_threshold * (0.75 if frame_brightness < 95.0 else 1.0)

            hand_label = "Unknown"
            if idx < len(handedness_entries) and handedness_entries[idx].classification:
                hand_label = handedness_entries[idx].classification[0].label

            hand_rank = 0 if hand_label == "Left" else 1 if hand_label == "Right" else 2
            hand_center_x = sum_x / max(len(hand_points), 1)

            if hand_span < span_threshold:
                rejected_entries.append((hand_span, hand_rank, hand_center_x, hand_points))
                continue

            hand_entries.append((hand_rank, hand_center_x, hand_points))

        if not hand_entries and rejected_entries:
            rejected_entries.sort(key=lambda item: item[0], reverse=True)
            best_span, best_rank, best_center_x, best_points = rejected_entries[0]
            if best_span >= max(0.018, span_threshold * 0.6):
                hand_entries.append((best_rank, best_center_x, best_points))

        hand_entries.sort(key=lambda item: (item[0], item[1]))
        hands_landmarks = [entry[2] for entry in hand_entries]

    if hands_landmarks:
        _last_hands_cache = hands_landmarks
        _last_hands_misses = 0
    elif _last_hands_cache and _last_hands_misses < HAND_CACHE_MAX_MISSES:
        hands_landmarks = _last_hands_cache
        _last_hands_misses += 1
    else:
        _last_hands_cache = []
        _last_hands_misses = 0

    if face_results is not None and face_results.multi_face_landmarks:
        for face in face_results.multi_face_landmarks:
            face_points = []
            for lm in face.landmark:
                face_points.append([lm.x, lm.y, lm.z])
            face_landmarks.append(face_points)

    return {
        "hands": hands_landmarks,
        "face": face_landmarks,
    }
