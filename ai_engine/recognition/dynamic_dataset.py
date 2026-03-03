import os
import json
import torch

MAX_HANDS = 2
HAND_POINTS = 21
FACE_POINTS = 468

def load_dynamic_dataset(base_path="recorded_sequences"):
    X = []
    y = []
    label_map = {}

    gestures = os.listdir(base_path)

    for idx, gesture in enumerate(gestures):
        label_map[gesture] = idx
        gesture_path = os.path.join(base_path, gesture)

        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)

            with open(file_path, "r") as f:
                sequence = json.load(f)

            flattened_sequence = []

            for frame in sequence:
                frame_features = []

                # Hands
                hands = frame["hands"]
                for i in range(MAX_HANDS):
                    if i < len(hands):
                        for point in hands[i]:
                            frame_features.extend(point)
                    else:
                        frame_features.extend([0.0] * HAND_POINTS * 3)

                # Face
                if len(frame["face"]) > 0:
                    for point in frame["face"][0]:
                        frame_features.extend(point)
                else:
                    frame_features.extend([0.0] * FACE_POINTS * 3)

                flattened_sequence.append(frame_features)

            X.append(flattened_sequence)
            y.append(idx)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y), label_map