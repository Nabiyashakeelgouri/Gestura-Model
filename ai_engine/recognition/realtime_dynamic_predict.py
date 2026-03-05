import torch
import torch.nn as nn
from ai_engine.preprocessing.landmark_extractor import extract_landmarks
from ai_engine.recognition.dynamic_dataset import MAX_HANDS, HAND_POINTS, FACE_POINTS

# -------- Label Map --------
label_map = {'hello': 0, 'no': 1, 'stop': 2, 'thank_u': 3, 'yes': 4}
reverse_map = {v: k for k, v in label_map.items()}

input_size = 1530
hidden_size = 128
num_classes = len(label_map)

# -------- Model --------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = LSTMModel()
model.load_state_dict(torch.load("models/dynamic_model.pth", map_location=torch.device("cpu")))
model.eval()

# -------- Sequence Buffer --------
sequence_buffer = []

def predict_dynamic(frame):

    global sequence_buffer

    landmarks = extract_landmarks(frame)

    frame_features = []

    hands = landmarks["hands"]

    for i in range(MAX_HANDS):
        if i < len(hands):
            for point in hands[i]:
                frame_features.extend(point)
        else:
            frame_features.extend([0.0] * HAND_POINTS * 3)

    if len(landmarks["face"]) > 0:
        for point in landmarks["face"][0]:
            frame_features.extend(point)
    else:
        frame_features.extend([0.0] * FACE_POINTS * 3)

    sequence_buffer.append(frame_features)

    # Wait until we have 30 frames
    if len(sequence_buffer) < 30:
        return "..."

    input_tensor = torch.tensor([sequence_buffer], dtype=torch.float32)

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence = torch.max(probabilities).item()

    predicted = torch.argmax(probabilities, dim=1).item()

    sequence_buffer = []

    if confidence > 0.60:
        return reverse_map[predicted]

    return "..."