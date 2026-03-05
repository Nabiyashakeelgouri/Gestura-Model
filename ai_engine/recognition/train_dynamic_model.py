import torch
import torch.nn as nn
import torch.optim as optim
from ai_engine.recognition.dynamic_dataset import load_dynamic_dataset

# Load data
X, y, label_map = load_dynamic_dataset()

input_size = X.shape[2]   # 1530
hidden_size = 128
num_classes = len(label_map)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last time step
        out = self.fc(out)
        return out

model = LSTMModel()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 20

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

print("Training complete")
torch.save(model.state_dict(), "models/dynamic_model.pth")
print("Model saved")