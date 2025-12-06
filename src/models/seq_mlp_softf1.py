import torch
import torch.nn as nn

from .base import BaseModel


def soft_f1_loss(y_pred, y_true, eps=1e-7):
    """
    y_pred: sigmoid output [B, M]
    y_true: {0,1} [B, M]
    """
    tp = (y_pred * y_true).sum(dim=0)
    fp = (y_pred * (1 - y_true)).sum(dim=0)
    fn = ((1 - y_pred) * y_true).sum(dim=0)
    soft_f1 = 2 * tp / (2 * tp + fp + fn + eps)
    return 1 - soft_f1.mean()


class SeqMLPSoftF1(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.3, bce_weight=0.5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss_soft = soft_f1_loss(probs, targets)
        loss_bce = self.bce(logits, targets)
        return self.bce_weight * loss_bce + (1 - self.bce_weight) * loss_soft
