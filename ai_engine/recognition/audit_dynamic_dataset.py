import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_FILE = ROOT_DIR / "datasets" / "dynamic" / "dynamic_data.json"
DEFAULT_DATASET_DIR = ROOT_DIR / "datasets" / "dynamic" / "recorded_sequences"
FALLBACK_DATASET_DIR = ROOT_DIR / "recorded_sequences"


def _resolve_dataset_source() -> Dict[str, Path]:
    env_file = os.getenv("DYNAMIC_DATA_FILE")
    if env_file:
        return {"type": "file", "path": Path(env_file)}
    if DEFAULT_DATA_FILE.exists():
        return {"type": "file", "path": DEFAULT_DATA_FILE}

    env_value = os.getenv("DYNAMIC_DATASET_DIR")
    if env_value:
        return {"type": "dir", "path": Path(env_value)}
    if DEFAULT_DATASET_DIR.exists():
        return {"type": "dir", "path": DEFAULT_DATASET_DIR}
    return {"type": "dir", "path": FALLBACK_DATASET_DIR}


def _hand_size(hand):
    xs = [point[0] for point in hand]
    ys = [point[1] for point in hand]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def _collect_items_from_dir(dataset_dir: Path) -> List[Dict[str, Any]]:
    if not dataset_dir.exists():
        return []

    nested = dataset_dir / "recorded_sequences"
    effective_dir = nested if nested.exists() else dataset_dir

    rows = []
    for cls in sorted([p for p in effective_dir.iterdir() if p.is_dir()]):
        for file_path in sorted([p for p in cls.iterdir() if p.suffix.lower() == ".json"]):
            try:
                sequence = json.loads(file_path.read_text(encoding="utf-8"))
                rows.append({"label": cls.name, "sequence": sequence})
            except Exception:
                continue
    return rows


def _collect_items_from_file(data_file: Path) -> List[Dict[str, Any]]:
    payload = json.loads(data_file.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("sequences"), list):
        return payload["sequences"]
    raise ValueError(f"Unsupported format in dynamic data file: {data_file}")


def run_audit():
    source = _resolve_dataset_source()
    source_type = source["type"]
    source_path = source["path"]

    if source_type == "file":
        if not source_path.exists():
            print(f"Dataset file not found: {source_path}")
            return
        rows = _collect_items_from_file(source_path)
    else:
        rows = _collect_items_from_dir(source_path)
        if not rows:
            print(f"Dataset not found or empty: {source_path}")
            return

    class_names = sorted(
        {
            str(item.get("label", "")).strip()
            for item in rows
            if str(item.get("label", "")).strip()
        }
    )
    print(f"Source: {source_path}")
    print(f"Classes: {class_names}")
    print(f"Class count: {len(class_names)}")
    print("")

    weak_classes = []
    for class_name in class_names:
        class_rows = [item for item in rows if str(item.get("label", "")).strip() == class_name]
        if not class_rows:
            print(f"{class_name}: no sequences")
            weak_classes.append(class_name)
            continue

        seq_lengths = []
        hand_ratios = []
        hand_sizes = []

        for row in class_rows:
            data = row.get("sequence", [])
            if not data:
                continue

            seq_lengths.append(len(data))
            hand_frames = 0
            per_seq_sizes = []

            for frame in data:
                hands = frame.get("hands", [])
                if hands:
                    hand_frames += 1
                    per_seq_sizes.append(_hand_size(hands[0]))

            hand_ratios.append(hand_frames / max(len(data), 1))
            if per_seq_sizes:
                hand_sizes.append(sum(per_seq_sizes) / len(per_seq_sizes))

        seq_count = len(class_rows)
        med_hand_ratio = median(hand_ratios) if hand_ratios else 0.0
        med_hand_size = median(hand_sizes) if hand_sizes else 0.0
        med_len = median(seq_lengths) if seq_lengths else 0

        print(
            f"{class_name}: seq={seq_count}, median_len={med_len}, "
            f"median_hand_ratio={med_hand_ratio:.2f}, median_hand_size={med_hand_size:.3f}"
        )

        if seq_count < 20 or med_hand_ratio < 0.75:
            weak_classes.append(class_name)

    print("")
    if weak_classes:
        print("Needs more/better data:")
        for name in weak_classes:
            print(f"- {name}")
    else:
        print("Dataset quality looks acceptable.")


if __name__ == "__main__":
    run_audit()
