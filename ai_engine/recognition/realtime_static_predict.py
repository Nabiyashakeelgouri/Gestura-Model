import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# -------- Load Classes --------
classes = sorted(os.listdir("datasets/asl_alphabet"))

# -------- Model --------
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
        x = self.fc(x)
        return x

model = StaticCNN(len(classes))
model.load_state_dict(torch.load("recognition/static_cnn_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop center square
    h, w, _ = frame.shape
    size = min(h, w)
    start_x = w // 2 - size // 2
    start_y = h // 2 - size // 2
    crop = frame[start_y:start_y+size, start_x:start_x+size]

    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probabilities = torch.softmax(output, dim=1)
        confidence = torch.max(probabilities).item()
        predicted = torch.argmax(probabilities, dim=1).item()

    if confidence > 0.70:
        text = f"{classes[predicted]} ({confidence:.2f})"
    else:
        text = "Low confidence"

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Static ASL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()