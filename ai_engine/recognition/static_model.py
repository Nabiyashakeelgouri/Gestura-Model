import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


HAND_POINTS = 21
TIP_INDICES = (4, 8, 12, 16, 20)
ANGLE_TRIPLETS = (
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
)
STATIC_FEATURE_SIZE = 108


def _safe_div(value: float, denom: float) -> float:
    return value / denom if abs(denom) > 1e-6 else 0.0


def _rotate_xy(x: float, y: float, angle: float) -> Tuple[float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return ((x * cos_a) - (y * sin_a), (x * sin_a) + (y * cos_a))


def _distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    return math.dist((float(p1[0]), float(p1[1]), float(p1[2])), (float(p2[0]), float(p2[1]), float(p2[2])))


def _joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ab = (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))
    cb = (float(c[0]) - float(b[0]), float(c[1]) - float(b[1]), float(c[2]) - float(b[2]))
    ab_norm = math.sqrt(sum(v * v for v in ab))
    cb_norm = math.sqrt(sum(v * v for v in cb))
    if ab_norm <= 1e-6 or cb_norm <= 1e-6:
        return 0.0
    cosine = sum(v1 * v2 for v1, v2 in zip(ab, cb)) / (ab_norm * cb_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return math.acos(cosine) / math.pi


def normalize_hand_points(hand: Sequence[Sequence[float]]) -> Optional[List[List[float]]]:
    if len(hand) != HAND_POINTS:
        return None

    points = [[float(point[0]), float(point[1]), float(point[2])] for point in hand]
    wrist = points[0]
    centered = [[point[0] - wrist[0], point[1] - wrist[1], point[2] - wrist[2]] for point in points]

    xs = [point[0] for point in centered]
    ys = [point[1] for point in centered]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3)

    palm_ref = centered[9]
    align_angle = -math.atan2(palm_ref[0], -palm_ref[1])

    rotated: List[List[float]] = []
    for x, y, z in centered:
        xr, yr = _rotate_xy(x, y, align_angle)
        rotated.append([_safe_div(xr, scale), _safe_div(yr, scale), _safe_div(z, scale)])

    if rotated[4][0] > rotated[20][0]:
        for point in rotated:
            point[0] *= -1.0

    return rotated


def extract_static_hand_features(hand: Sequence[Sequence[float]]) -> Optional[List[float]]:
    normalized = normalize_hand_points(hand)
    if normalized is None:
        return None

    features: List[float] = []
    for point in normalized:
        features.extend(point)

    for idx in TIP_INDICES:
        features.append(_distance(normalized[0], normalized[idx]))

    tip_pairs = (
        (4, 8),
        (4, 12),
        (4, 16),
        (4, 20),
        (8, 12),
        (8, 16),
        (8, 20),
        (12, 16),
        (12, 20),
        (16, 20),
    )
    for a_idx, b_idx in tip_pairs:
        features.append(_distance(normalized[a_idx], normalized[b_idx]))

    for a_idx, b_idx, c_idx in ANGLE_TRIPLETS:
        features.append(_joint_angle(normalized[a_idx], normalized[b_idx], normalized[c_idx]))

    direction_pairs = (
        (4, 2),
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17),
    )
    for tip_idx, base_idx in direction_pairs:
        direction = [
            normalized[tip_idx][axis] - normalized[base_idx][axis]
            for axis in range(3)
        ]
        length = math.sqrt(sum(value * value for value in direction))
        if length <= 1e-6:
            features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([value / length for value in direction])

    if len(features) != STATIC_FEATURE_SIZE:
        raise ValueError(f"Expected {STATIC_FEATURE_SIZE} static features, got {len(features)}")

    return features


def build_static_feature_vector(landmarks: Dict[str, List]) -> Optional[List[float]]:
    hands = landmarks.get("hands", [])
    if not hands:
        return None

    hand = hands[0]
    if len(hand) != HAND_POINTS:
        return None

    return extract_static_hand_features(hand)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class StaticGestureModel(nn.Module):
    def __init__(self, num_classes: int, input_size: int = STATIC_FEATURE_SIZE):
        super().__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.12),
            ResidualMLPBlock(256, dropout=0.12),
            ResidualMLPBlock(256, dropout=0.12),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
