import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ai_engine.recognition.static_model import STATIC_FEATURE_SIZE, StaticGestureModel, extract_static_hand_features


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / 'datasets' / 'static' / 'asl_alphabet'
MODEL_PATH = ROOT_DIR / 'models' / 'static_cnn_model.pth'
CLASS_MAP_PATH = ROOT_DIR / 'models' / 'static_class_map.json'
REPORT_PATH = ROOT_DIR / 'models' / 'static_training_report.json'
CACHE_PATH = ROOT_DIR / 'models' / 'static_landmark_cache.pt'

BATCH_SIZE = 64
EPOCHS = 120
VAL_SPLIT = 0.2
SEED = 42
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4
EARLY_STOPPING_PATIENCE = 18
LABEL_SMOOTHING = 0.05
DETECTION_CONFIDENCE = 0.35


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.features = features.float()
        self.labels = labels.long()
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        features = self.features[idx].clone()
        if self.augment:
            features[:63] += torch.randn(63, dtype=features.dtype) * 0.01
            features[63:78] += torch.randn(15, dtype=features.dtype) * 0.006
            features[78:] += torch.randn(features.numel() - 78, dtype=features.dtype) * 0.004
        return features, self.labels[idx]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _dataset_signature(image_paths: Sequence[Path]) -> Dict[str, float]:
    latest_mtime = max((path.stat().st_mtime for path in image_paths), default=0.0)
    return {
        'count': len(image_paths),
        'latest_mtime': round(float(latest_mtime), 3),
    }


def _load_cached_features(signature: Dict[str, float]) -> Optional[Dict[str, object]]:
    if not CACHE_PATH.exists():
        return None
    cache = torch.load(CACHE_PATH, map_location='cpu')
    if cache.get('signature') != signature:
        return None
    return cache


def _extract_features_from_image(image_path: Path, hands) -> Optional[List[float]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    candidates = [image]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 90.0:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        candidates.append(cv2.convertScaleAbs(enhanced, alpha=1.12, beta=10))

    for candidate in candidates:
        rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            hand = [[lm.x, lm.y, lm.z] for lm in result.multi_hand_landmarks[0].landmark]
            features = extract_static_hand_features(hand)
            if features is not None:
                return features
    return None


def _build_feature_cache() -> Dict[str, object]:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f'Dataset not found: {DATASET_DIR}')

    image_paths = sorted(DATASET_DIR.rglob('*.jpg'))
    if not image_paths:
        raise FileNotFoundError(f'No JPG files found in: {DATASET_DIR}')

    signature = _dataset_signature(image_paths)
    cached = _load_cached_features(signature)
    if cached is not None:
        return cached

    labels = sorted([path.name for path in DATASET_DIR.iterdir() if path.is_dir()])
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    features: List[List[float]] = []
    targets: List[int] = []
    kept_paths: List[str] = []
    skipped_paths: List[str] = []
    per_class_total = {label: 0 for label in labels}
    per_class_kept = {label: 0 for label in labels}

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
    ) as hands:
        for image_path in image_paths:
            label = image_path.parent.name
            per_class_total[label] += 1
            feature_vector = _extract_features_from_image(image_path, hands)
            if feature_vector is None:
                skipped_paths.append(str(image_path))
                continue
            features.append(feature_vector)
            targets.append(label_to_idx[label])
            kept_paths.append(str(image_path))
            per_class_kept[label] += 1

    cache = {
        'signature': signature,
        'labels': labels,
        'features': torch.tensor(features, dtype=torch.float32),
        'targets': torch.tensor(targets, dtype=torch.long),
        'kept_paths': kept_paths,
        'skipped_paths': skipped_paths,
        'per_class_total': per_class_total,
        'per_class_kept': per_class_kept,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE_PATH)
    return cache


def _stratified_split(labels: Sequence[int]) -> Tuple[List[int], List[int]]:
    by_label: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(idx)

    rng = random.Random(SEED)
    train_idx: List[int] = []
    val_idx: List[int] = []

    for indices in by_label.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * VAL_SPLIT)))
        val_count = min(val_count, len(shuffled) - 1)
        val_idx.extend(shuffled[:val_count])
        train_idx.extend(shuffled[val_count:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> Dict[str, object]:
    num_classes = len(labels)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth)][int(pred)] += 1

    per_class = {}
    recalls = []
    for idx, label in enumerate(labels):
        row_total = sum(confusion[idx])
        correct = confusion[idx][idx]
        recall = (correct / row_total) if row_total else 0.0
        recalls.append(recall)
        per_class[label] = {
            'samples': row_total,
            'correct': correct,
            'accuracy': round(recall, 4),
        }

    overall = 0.0
    if y_true:
        overall = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)

    return {
        'accuracy': round(overall, 4),
        'macro_accuracy': round(sum(recalls) / max(len(recalls), 1), 4),
        'per_class': per_class,
        'confusion_matrix': confusion,
    }


def _run_eval(model: nn.Module, loader: DataLoader, labels: Sequence[str], device: torch.device) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device)
            logits = model(features)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(targets.tolist())
    return _evaluate_predictions(y_true, y_pred, labels)


def train() -> None:
    _set_seed(SEED)

    cache = _build_feature_cache()
    labels = list(cache['labels'])
    features = cache['features']
    targets = cache['targets']

    if int(targets.numel()) == 0:
        raise RuntimeError('Static landmark extraction produced zero usable samples.')

    train_idx, val_idx = _stratified_split(targets.tolist())
    train_features = features[train_idx]
    train_targets = targets[train_idx]
    val_features = features[val_idx]
    val_targets = targets[val_idx]

    train_dataset = FeatureDataset(train_features, train_targets, augment=True)
    val_dataset = FeatureDataset(val_features, val_targets, augment=False)
    train_eval_dataset = FeatureDataset(train_features, train_targets, augment=False)
    full_dataset = FeatureDataset(features, targets, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StaticGestureModel(num_classes=len(labels), input_size=STATIC_FEATURE_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    best_state = None
    best_val_acc = -1.0
    best_val_macro = -1.0
    best_epoch = 0
    stale_epochs = 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_features)
            loss = criterion(logits, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_features.size(0))
            total_correct += int((torch.argmax(logits, dim=1) == batch_targets).sum().item())
            total_seen += int(batch_features.size(0))

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        val_metrics = _run_eval(model, val_loader, labels, device)
        val_acc = float(val_metrics['accuracy'])
        val_macro = float(val_metrics['macro_accuracy'])

        print(
            f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} val_macro={val_macro:.3f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        improved = (val_macro > best_val_macro) or (val_macro == best_val_macro and val_acc >= best_val_acc)
        if improved:
            best_val_acc = val_acc
            best_val_macro = val_macro
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    'model_type': 'static_landmark_mlp',
                    'input_size': STATIC_FEATURE_SIZE,
                    'num_classes': len(labels),
                    'state_dict': best_state,
                },
                MODEL_PATH,
            )
        else:
            stale_epochs += 1

        scheduler.step(val_macro)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping at epoch {epoch}.')
            break

    if best_state is None:
        raise RuntimeError('Static training did not produce a valid model state.')

    model.load_state_dict(best_state)
    model.to(device)

    train_metrics = _run_eval(model, train_eval_loader, labels, device)
    val_metrics = _run_eval(model, val_loader, labels, device)
    all_metrics = _run_eval(model, full_loader, labels, device)

    CLASS_MAP_PATH.write_text(json.dumps(labels, ensure_ascii=True, indent=2), encoding='utf-8')
    REPORT_PATH.write_text(
        json.dumps(
            {
                'model_type': 'static_landmark_mlp',
                'feature_size': STATIC_FEATURE_SIZE,
                'best_epoch': best_epoch,
                'best_val_accuracy': best_val_acc,
                'best_val_macro_accuracy': best_val_macro,
                'usable_samples': int(targets.numel()),
                'skipped_samples': len(cache['skipped_paths']),
                'train_samples': len(train_idx),
                'val_samples': len(val_idx),
                'labels': labels,
                'landmark_coverage': {
                    'kept': int(targets.numel()),
                    'total': int(targets.numel()) + len(cache['skipped_paths']),
                    'ratio': round(int(targets.numel()) / max(int(targets.numel()) + len(cache['skipped_paths']), 1), 4),
                    'per_class_total': cache['per_class_total'],
                    'per_class_kept': cache['per_class_kept'],
                },
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'all_metrics': all_metrics,
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    print('Training complete.')
    print(f'Best epoch: {best_epoch}')
    print(f'Best val_acc={best_val_acc:.3f} val_macro={best_val_macro:.3f}')
    print(f'Usable samples: {int(targets.numel())} / {int(targets.numel()) + len(cache["skipped_paths"])}')
    print('Per-class validation accuracy:')
    for label in labels:
        metrics = val_metrics['per_class'].get(label, {})
        print(f"- {label}: {metrics.get('accuracy', 0.0):.3f} ({metrics.get('correct', 0)}/{metrics.get('samples', 0)})")
    print(f'Saved model: {MODEL_PATH}')
    print(f'Saved class map: {CLASS_MAP_PATH}')
    print(f'Saved report: {REPORT_PATH}')


if __name__ == '__main__':
    train()
