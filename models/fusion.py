import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """简单的文本-图像特征融合模块。

    目前使用特征拼接（concatenation）+ 多层感知机（MLP）。
    后续可以尝试更高级的融合方式，比如注意力、门控机制、双线性池化等。
    """

    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """参数:
        - text_feat: (B, text_dim)，文本模态特征
        - image_feat: (B, image_dim)，图像模态特征

        返回:
        - fused: (B, output_dim)，融合后的多模态特征
        """
        x = torch.cat([text_feat, image_feat], dim=-1)
        return self.fc(x)
