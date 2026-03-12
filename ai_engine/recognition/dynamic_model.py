import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ai_engine.recognition.dynamic_dataset import DYNAMIC_INPUT_SIZE


ROOT_DIR = Path(__file__).resolve().parents[2]
LABEL_MAP_PATH = ROOT_DIR / "models" / "dynamic_label_map.json"



def load_dynamic_label_map() -> Dict[str, int]:
    if LABEL_MAP_PATH.exists():
        payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(label): int(index) for label, index in payload.items()}
    return {}


class DynamicGestureModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = DYNAMIC_INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.feature_norm = nn.LayerNorm(input_size * 2)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        features = torch.cat([x, delta], dim=-1)
        features = self.feature_norm(features)
        encoded = self.input_proj(features)

        out, _ = self.lstm(encoded)
        attn_scores = torch.softmax(self.attn(out).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = torch.sum(out * attn_scores, dim=1)
        tail = out[:, -1, :]
        return self.head(torch.cat([pooled, tail], dim=1))

