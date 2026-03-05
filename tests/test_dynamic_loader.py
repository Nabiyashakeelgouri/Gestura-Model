from ai_engine.recognition.dynamic_dataset import load_dynamic_dataset

X, y, label_map = load_dynamic_dataset()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", label_map)