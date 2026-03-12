import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import cv2


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = ROOT_DIR / "datasets" / "dynamic" / "dynamic_data.json"
DEFAULT_LABELS = ["hello", "no", "stop", "thank_u", "yes"]


def _load_dynamic_payload(output_file: Path) -> dict:
    if not output_file.exists():
        return {
            "version": 1,
            "description": "Consolidated dynamic landmark dataset",
            "sequence_count": 0,
            "class_count": 0,
            "label_counts": {},
            "sequences": [],
        }

    try:
        payload = json.loads(output_file.read_text(encoding="utf-8"))
    except Exception:
        return {
            "version": 1,
            "description": "Consolidated dynamic landmark dataset",
            "sequence_count": 0,
            "class_count": 0,
            "label_counts": {},
            "sequences": [],
        }

    if isinstance(payload, dict) and isinstance(payload.get("sequences"), list):
        payload.setdefault("label_counts", {})
        payload.setdefault("version", 1)
        payload.setdefault("description", "Consolidated dynamic landmark dataset")
        return payload

    # Backward-compatible conversion if file is only a list.
    if isinstance(payload, list):
        return {
            "version": 1,
            "description": "Consolidated dynamic landmark dataset",
            "sequence_count": len(payload),
            "class_count": 0,
            "label_counts": {},
            "sequences": payload,
        }

    return {
        "version": 1,
        "description": "Consolidated dynamic landmark dataset",
        "sequence_count": 0,
        "class_count": 0,
        "label_counts": {},
        "sequences": [],
    }


def _save_dynamic_payload(output_file: Path, payload: dict) -> None:
    payload["sequence_count"] = len(payload.get("sequences", []))
    payload["class_count"] = len([k for k, v in payload.get("label_counts", {}).items() if v > 0])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _next_sequence_index_for_label(payload: dict, label: str) -> int:
    sequence_indexes = []
    for row in payload.get("sequences", []):
        if str(row.get("label", "")).strip() != label:
            continue
        value = row.get("sequence_id")
        if isinstance(value, int):
            sequence_indexes.append(value)
    return (max(sequence_indexes) + 1) if sequence_indexes else 0


def _draw_overlay(
    frame,
    label: str,
    current_seq: int,
    total_seq: int,
    frames_collected: int,
    target_frames: int,
    countdown_text: str,
):
    h, _ = frame.shape[:2]
    cv2.putText(frame, f"Gesture: {label}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Sequence: {current_seq}/{total_seq}",
        (16, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Frames: {frames_collected}/{target_frames}",
        (16, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "Keep hand clearly visible. Press Q to quit, N to skip label.",
        (16, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    if countdown_text:
        cv2.putText(
            frame,
            countdown_text,
            (16, 146),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 215, 255),
            2,
        )


def _open_camera(camera_index: int, startup_timeout: float) -> cv2.VideoCapture:
    candidates = []
    for idx in [camera_index, 0, 1, 2]:
        if idx not in candidates:
            candidates.append(idx)

    for idx in candidates:
        # DirectShow backend usually starts faster and more reliably on Windows.
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            cap.release()
            continue

        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            ok, _ = cap.read()
            if ok:
                print(f"Camera opened on index {idx}")
                return cap
            time.sleep(0.05)

        cap.release()

    raise RuntimeError(
        "Unable to start camera. Close browser tabs/video apps using webcam, then retry "
        "with --camera-index 0 or --camera-index 1."
    )


def collect_sequences(
    labels: List[str],
    sequences_per_label: int,
    sequence_len: int,
    prep_seconds: float,
    rest_seconds: float,
    min_hand_frames: int,
    camera_index: int,
    startup_timeout: float,
) -> None:
    cap = _open_camera(camera_index=camera_index, startup_timeout=startup_timeout)
    from ai_engine.preprocessing.landmark_extractor import extract_landmarks
    payload = _load_dynamic_payload(OUTPUT_FILE)

    cv2.namedWindow("Gestura Dynamic Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gestura Dynamic Collector", 1100, 700)

    try:
        for label in labels:
            seq_idx = _next_sequence_index_for_label(payload, label)
            completed = 0

            while completed < sequences_per_label:
                sequence = []
                hand_presence = []

                prep_end = time.monotonic() + prep_seconds
                skip_label = False
                while time.monotonic() < prep_end:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    countdown = max(0.0, prep_end - time.monotonic())
                    _draw_overlay(
                        frame=frame,
                        label=label,
                        current_seq=completed + 1,
                        total_seq=sequences_per_label,
                        frames_collected=0,
                        target_frames=sequence_len,
                        countdown_text=f"Get ready: {countdown:.1f}s",
                    )
                    cv2.imshow("Gestura Dynamic Collector", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                    if key == ord("n"):
                        skip_label = True
                        break
                if skip_label:
                    break

                while len(sequence) < sequence_len:
                    ok, frame = cap.read()
                    if not ok:
                        continue

                    landmarks = extract_landmarks(frame)
                    sequence.append(landmarks)
                    hand_presence.append(1 if landmarks.get("hands") else 0)

                    _draw_overlay(
                        frame=frame,
                        label=label,
                        current_seq=completed + 1,
                        total_seq=sequences_per_label,
                        frames_collected=len(sequence),
                        target_frames=sequence_len,
                        countdown_text="Recording...",
                    )
                    cv2.imshow("Gestura Dynamic Collector", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        return
                    if key == ord("n"):
                        skip_label = True
                        break

                if skip_label:
                    break

                if sum(hand_presence) < min_hand_frames:
                    retry_until = time.monotonic() + rest_seconds
                    while time.monotonic() < retry_until:
                        ok, frame = cap.read()
                        if not ok:
                            continue
                        _draw_overlay(
                            frame=frame,
                            label=label,
                            current_seq=completed + 1,
                            total_seq=sequences_per_label,
                            frames_collected=0,
                            target_frames=sequence_len,
                            countdown_text=(
                                f"Low hand visibility ({sum(hand_presence)}/{sequence_len}). "
                                "Retrying..."
                            ),
                        )
                        cv2.imshow("Gestura Dynamic Collector", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord("q"):
                            return
                    continue

                payload.setdefault("sequences", []).append(
                    {
                        "label": label,
                        "sequence_id": seq_idx,
                        "sequence": sequence,
                        "source_file": f"dynamic_data:{label}:seq_{seq_idx}",
                    }
                )
                counts = payload.setdefault("label_counts", {})
                counts[label] = int(counts.get(label, 0)) + 1
                _save_dynamic_payload(OUTPUT_FILE, payload)

                seq_idx += 1
                completed += 1

                rest_end = time.monotonic() + rest_seconds
                while time.monotonic() < rest_end:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    _draw_overlay(
                        frame=frame,
                        label=label,
                        current_seq=completed,
                        total_seq=sequences_per_label,
                        frames_collected=sequence_len,
                        target_frames=sequence_len,
                        countdown_text=(
                            f"Saved to dynamic_data.json. Next starts in "
                            f"{max(0.0, rest_end - time.monotonic()):.1f}s"
                        ),
                    )
                    cv2.imshow("Gestura Dynamic Collector", frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        return

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Guided dynamic sequence collector for Gestura.",
    )
    parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated gesture labels in collection order.",
    )
    parser.add_argument("--sequences-per-label", type=int, default=25)
    parser.add_argument("--sequence-len", type=int, default=30)
    parser.add_argument("--prep-seconds", type=float, default=2.0)
    parser.add_argument("--rest-seconds", type=float, default=1.0)
    parser.add_argument("--min-hand-frames", type=int, default=20)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=6.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required.")

    if args.min_hand_frames > args.sequence_len:
        raise ValueError("--min-hand-frames must be <= --sequence-len")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    collect_sequences(
        labels=labels,
        sequences_per_label=args.sequences_per_label,
        sequence_len=args.sequence_len,
        prep_seconds=args.prep_seconds,
        rest_seconds=args.rest_seconds,
        min_hand_frames=args.min_hand_frames,
        camera_index=args.camera_index,
        startup_timeout=args.startup_timeout,
    )
