import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """使用跨模态注意力（Cross-Modal Attention）的文本-图像特征融合模块。
    """

    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int = 256, dropout: float = 0.1, num_heads: int = 4):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        q = self.text_proj(text_feat).unsqueeze(1)
        k = self.image_proj(image_feat).unsqueeze(1)
        v = k

        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        fused = attn_output.squeeze(1)

        fused = self.ffn(fused)
        return fused
