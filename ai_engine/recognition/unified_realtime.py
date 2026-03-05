import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import os
import uuid
import pygame
from gtts import gTTS

from ai_engine.preprocessing.landmark_extractor import extract_landmarks
from ai_engine.recognition.dynamic_dataset import MAX_HANDS, HAND_POINTS, FACE_POINTS
from ai_engine.nlp.sentence_builder import correct_sentence


# ---------------- TTS ----------------
last_speech_time = 0

def speak(text):
    global last_speech_time
    current_time = time.time()

    if current_time - last_speech_time > 1.5:
        try:
            filename = f"speech_{uuid.uuid4().hex}.mp3"

            tts = gTTS(text=str(text), lang="en")
            tts.save(filename)

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            last_speech_time = current_time

        except Exception as e:
            print("TTS Error:", e)


# ---------------- STATIC MODEL ----------------
static_classes = sorted(os.listdir("datasets/static/asl_alphabet"))

class StaticCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


static_model = StaticCNN(len(static_classes))
static_model.load_state_dict(
    torch.load("models/static_cnn_model.pth", map_location="cpu")
)
static_model.eval()

static_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# ---------------- DYNAMIC MODEL ----------------
dynamic_labels = {"hello":0,"no":1,"stop":2,"thank_u":3,"yes":4}
reverse_dynamic = {v:k for k,v in dynamic_labels.items()}

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(1530,128,batch_first=True)
        self.fc = nn.Linear(128,len(dynamic_labels))

    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc(out)


dynamic_model = LSTMModel()
dynamic_model.load_state_dict(
    torch.load("models/dynamic_model.pth", map_location="cpu")
)
dynamic_model.eval()


# ---------------- MAIN CAMERA FUNCTION ----------------
def predict_gesture():

    pygame.mixer.init()

    cap = cv2.VideoCapture(0)

    sequence = []
    sentence = []

    last_static = ""
    last_static_time = 0
    last_dynamic_time = 0

    active = False
    activation_timer = None
    deactivation_timer = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        hands_detected = len(landmarks["hands"]) > 0

        # ---------- Activation ----------
        if hands_detected and not active:
            if activation_timer is None:
                activation_timer = time.time()

            elif time.time() - activation_timer > 2:
                active = True
                speak("Dynamic mode")

        else:
            activation_timer = None

        # ---------- Deactivation ----------
        if not hands_detected and active:
            if deactivation_timer is None:
                deactivation_timer = time.time()

            elif time.time() - deactivation_timer > 2:
                active = False
                sequence.clear()
                speak("Static mode")

        else:
            deactivation_timer = None


        display_text = ""


        # ================= DYNAMIC =================
        if active:

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

            if len(sequence) == 30:

                input_tensor = torch.tensor([sequence], dtype=torch.float32)

                with torch.no_grad():
                    output = dynamic_model(input_tensor)
                    probs = torch.softmax(output, dim=1)

                    conf = torch.max(probs).item()
                    pred = torch.argmax(probs, dim=1).item()

                detected_word = None

                if conf > 0.45:
                    detected_word = reverse_dynamic[pred]

                current_time = time.time()

                if detected_word and current_time - last_dynamic_time > 1.5:

                    sentence.append(detected_word)
                    speak(" ".join(sentence))

                    last_dynamic_time = current_time

                sequence.clear()

            clean_sentence = correct_sentence(sentence)

            display_text = "Dynamic: " + clean_sentence


        # ================= STATIC =================
        else:

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = static_transform(img).unsqueeze(0)

            with torch.no_grad():

                output = static_model(img)

                probs = torch.softmax(output, dim=1)

                conf = torch.max(probs).item()
                pred = torch.argmax(probs, dim=1).item()

            if conf > 0.65:

                current_static = static_classes[pred]

                current_time = time.time()

                if current_time - last_static_time > 1.5:
                    speak(current_static)
                    last_static_time = current_time

                last_static = current_static

            display_text = "Static: " + last_static


        cv2.putText(
            frame,
            display_text,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Gestura Unified System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()