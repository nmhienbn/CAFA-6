from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
            return torch.sigmoid(out)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location="cpu"):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
