import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------- Transforms --------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# -------- Load Dataset --------
dataset = datasets.ImageFolder("datasets/static/asl_alphabet", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_classes = len(dataset.classes)

# -------- CNN Model --------
class StaticCNN(nn.Module):
    def __init__(self):
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

model = StaticCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------- Training --------
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# -------- Save Model --------
torch.save(model.state_dict(), "models/static_cnn_model.pth")
print("Static CNN Model Saved")