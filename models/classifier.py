import torch.nn as nn


class Classifier(nn.Module):
    """用于三分类情感分类任务的简单分类头。"""

    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features)
