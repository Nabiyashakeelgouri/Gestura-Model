from fastapi import APIRouter
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from ai_engine.recognition.unified_realtime import static_model, static_classes

router = APIRouter()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

@router.get("/predict-static")
def predict_static():

    img_path = "tests/test.jpg"   # temporary test image

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = static_model(img)
        probs = torch.softmax(output, dim=1)
        conf = torch.max(probs).item()
        pred = torch.argmax(probs, dim=1).item()

    label = static_classes[pred]

    return {
        "gesture": label,
        "confidence": conf
    }