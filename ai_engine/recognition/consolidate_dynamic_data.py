import argparse
import json
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT_DIR / "datasets" / "dynamic" / "recorded_sequences"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "datasets" / "dynamic" / "dynamic_data.json"


def _resolve_effective_input_dir(input_dir: Path) -> Path:
    nested = input_dir / "recorded_sequences"
    return nested if nested.exists() else input_dir


def consolidate(input_dir: Path, output_file: Path) -> Dict[str, int]:
    effective_dir = _resolve_effective_input_dir(input_dir)
    if not effective_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {effective_dir}")

    sequences: List[Dict] = []
    label_counts: Dict[str, int] = {}

    class_dirs = sorted([p for p in effective_dir.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        label = class_dir.name
        files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() == ".json"])
        label_counts[label] = len(files)

        for file_path in files:
            try:
                sequence = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            sequences.append(
                {
                    "label": label,
                    "sequence": sequence,
                    "source_file": str(file_path.relative_to(ROOT_DIR)),
                }
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "description": "Consolidated dynamic landmark dataset",
        "sequence_count": len(sequences),
        "class_count": len([name for name, count in label_counts.items() if count > 0]),
        "label_counts": label_counts,
        "sequences": sequences,
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return label_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Consolidate dynamic sequences into one JSON file.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    counts = consolidate(input_dir=input_dir, output_file=output_file)
    print(f"Saved consolidated dynamic data: {output_file}")
    print("Label counts:")
    for label, count in sorted(counts.items()):
        print(f"- {label}: {count}")
