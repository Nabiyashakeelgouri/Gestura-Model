import cv2
import torch
import torch.nn as nn
import time
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
model.load_state_dict(torch.load("recognition/dynamic_model.pth"))
model.eval()

# -------- Camera --------
cap = cv2.VideoCapture(0)

sequence = []
sentence = []
last_prediction = None

active = False
activation_timer = None
deactivation_timer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame)

    # -------- Activation Logic --------
    hands_detected = len(landmarks["hands"]) > 0

    if hands_detected and not active:
        if activation_timer is None:
            activation_timer = time.time()
        elif time.time() - activation_timer > 2:
            active = True
            print("Activated")
    else:
        activation_timer = None

    if not hands_detected and active:
        if deactivation_timer is None:
            deactivation_timer = time.time()
        elif time.time() - deactivation_timer > 2:
            active = False
            print("Deactivated")
            sentence.clear()
            last_prediction = None
    else:
        deactivation_timer = None

    # -------- Feature Extraction --------
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

    sequence.append(frame_features)

    # -------- Prediction --------
    if active and len(sequence) == 30:
        input_tensor = torch.tensor([sequence], dtype=torch.float32)
        output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        predicted = torch.argmax(probabilities, dim=1).item()

        if confidence > 0.60:
            current_word = reverse_map[predicted]

            if last_prediction is None or current_word != last_prediction:
                sentence.append(current_word)
                print("Sentence:", " ".join(sentence))
                last_prediction = current_word

        sequence = []

    cv2.imshow("Dynamic Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()