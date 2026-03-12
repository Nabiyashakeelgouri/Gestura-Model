import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from ai_engine.recognition.dynamic_dataset import load_dynamic_dataset
from ai_engine.recognition.dynamic_model import DynamicGestureModel


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "dynamic_model.pth"
LABEL_MAP_PATH = ROOT_DIR / "models" / "dynamic_label_map.json"
REPORT_PATH = ROOT_DIR / "models" / "dynamic_training_report.json"

VAL_SPLIT = float(os.getenv("DYNAMIC_VAL_SPLIT", "0.2"))
BATCH_SIZE = int(os.getenv("DYNAMIC_BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("DYNAMIC_EPOCHS", "60"))
LEARNING_RATE = float(os.getenv("DYNAMIC_LEARNING_RATE", "0.0007"))
MIN_HAND_FRAMES_RATIO = float(os.getenv("DYNAMIC_MIN_HAND_FRAMES_RATIO", "0.4"))
WEIGHT_DECAY = float(os.getenv("DYNAMIC_WEIGHT_DECAY", "0.0001"))
SEED = int(os.getenv("DYNAMIC_TRAIN_SEED", "42"))
EARLY_STOPPING_PATIENCE = int(os.getenv("DYNAMIC_EARLY_STOPPING_PATIENCE", "15"))

AUGMENT_TRAIN = os.getenv("DYNAMIC_AUGMENT", "true").strip().lower() in {"1", "true", "yes", "on"}
AUG_NOISE_STD = float(os.getenv("DYNAMIC_AUG_NOISE_STD", "0.008"))
AUG_SHIFT_MAX = int(os.getenv("DYNAMIC_AUG_SHIFT_MAX", "2"))
AUG_FRAME_DROP_P = float(os.getenv("DYNAMIC_AUG_FRAME_DROP_P", "0.04"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class DynamicSequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _augment_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        out = seq.clone()

        if random.random() < 0.8 and AUG_NOISE_STD > 0:
            out = out + torch.randn_like(out) * AUG_NOISE_STD

        if AUG_SHIFT_MAX > 0 and random.random() < 0.6:
            shift = random.randint(-AUG_SHIFT_MAX, AUG_SHIFT_MAX)
            if shift > 0:
                out = torch.cat([out[shift:], out[-1:].repeat(shift, 1)], dim=0)
            elif shift < 0:
                s = -shift
                out = torch.cat([out[:1].repeat(s, 1), out[:-s]], dim=0)

        if AUG_FRAME_DROP_P > 0 and random.random() < 0.7:
            for t in range(1, out.shape[0]):
                if random.random() < AUG_FRAME_DROP_P:
                    out[t] = out[t - 1]

        return out

    def __getitem__(self, idx: int):
        seq = self.X[idx]
        if self.augment:
            seq = self._augment_sequence(seq)
        return seq, self.y[idx]


def _stratified_split_indices(y: torch.Tensor) -> Tuple[List[int], List[int]]:
    total = int(y.shape[0])
    if total < 2:
        raise ValueError("Need at least 2 dynamic sequences for train/validation split.")

    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(y.tolist()):
        label_to_indices[int(label)].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    rng = random.Random(SEED)

    for indices in label_to_indices.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            train_idx.extend(shuffled)
            continue

        val_count = max(1, int(round(len(shuffled) * VAL_SPLIT)))
        val_count = min(val_count, len(shuffled) - 1)

        val_idx.extend(shuffled[:val_count])
        train_idx.extend(shuffled[val_count:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if not train_idx or not val_idx:
        raise ValueError("Stratified split failed to produce both train and validation samples.")

    return train_idx, val_idx


def _make_loaders(X: torch.Tensor, y: torch.Tensor) -> Tuple[DataLoader, DataLoader, List[int], List[int]]:
    train_idx, val_idx = _stratified_split_indices(y)

    train_dataset = DynamicSequenceDataset(X, y, augment=AUGMENT_TRAIN)
    val_dataset = DynamicSequenceDataset(X, y, augment=False)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, train_idx, val_idx


def _compute_class_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float()
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights


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
            "samples": row_total,
            "correct": correct,
            "accuracy": round(recall, 4),
        }

    overall = 0.0
    if y_true:
        overall = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)

    return {
        "accuracy": round(overall, 4),
        "macro_accuracy": round(sum(recalls) / max(len(recalls), 1), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _run_eval(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
    labels: Sequence[str],
) -> Dict[str, object]:
    if not indices:
        return {"accuracy": 0.0, "macro_accuracy": 0.0, "per_class": {}, "confusion_matrix": []}

    subset_x = X[list(indices)]
    subset_y = y[list(indices)]

    model.eval()
    with torch.inference_mode():
        logits = model(subset_x)
        preds = torch.argmax(logits, dim=1)

    return _evaluate_predictions(subset_y.tolist(), preds.tolist(), labels)


def train() -> None:
    _set_seed(SEED)

    X, y, label_map = load_dynamic_dataset(min_hand_frames_ratio=MIN_HAND_FRAMES_RATIO)
    labels = [label for label, _ in sorted(label_map.items(), key=lambda item: int(item[1]))]
    num_classes = len(label_map)

    train_loader, val_loader, train_idx, val_idx = _make_loaders(X, y)

    model = DynamicGestureModel(num_classes=num_classes)
    class_weights = _compute_class_weights(y, num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6,
        min_lr=1e-5,
    )

    best_state = None
    best_val_macro = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    stale_epochs = 0

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item()) * int(batch_x.size(0))
            preds = torch.argmax(logits, dim=1)
            train_correct += int((preds == batch_y).sum().item())
            train_total += int(batch_x.size(0))

        val_metrics = _run_eval(model, X, y, val_idx, labels)
        train_loss_avg = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = float(val_metrics["accuracy"])
        val_macro = float(val_metrics["macro_accuracy"])

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss_avg:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} val_macro={val_macro:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        improved = (val_macro > best_val_macro) or (val_macro == best_val_macro and val_acc >= best_val_acc)
        if improved:
            best_val_macro = val_macro
            best_val_acc = val_acc
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, MODEL_PATH)
        else:
            stale_epochs += 1

        scheduler.step(val_macro)

        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch} after {stale_epochs} stale epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)

    train_metrics = _run_eval(model, X, y, train_idx, labels)
    val_metrics = _run_eval(model, X, y, val_idx, labels)
    all_metrics = _run_eval(model, X, y, list(range(int(X.shape[0]))), labels)

    report = {
        "seed": SEED,
        "epochs_requested": EPOCHS,
        "best_epoch": best_epoch,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "labels": labels,
        "best_val_accuracy": best_val_acc,
        "best_val_macro_accuracy": best_val_macro,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "all_metrics": all_metrics,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_acc={best_val_acc:.3f} val_macro={best_val_macro:.3f}")
    print("Per-class validation accuracy:")
    for label in labels:
        class_metrics = val_metrics["per_class"].get(label, {})
        print(
            f"- {label}: {class_metrics.get('accuracy', 0.0):.3f} "
            f"({class_metrics.get('correct', 0)}/{class_metrics.get('samples', 0)})"
        )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved label map: {LABEL_MAP_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    train()
