import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """使用跨模态注意力（Cross-Modal Attention）的文本-图像特征融合模块。

    当前实现采用「文本作为 Query，图像作为 Key/Value」的多头注意力，
    然后对得到的融合向量再做一层前馈网络。
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

        # 投影到同一维度，构造长度为 1 的序列，便于使用 MultiheadAttention
        # 形状均为 (B, 1, hidden_dim)
        q = self.text_proj(text_feat).unsqueeze(1)
        k = self.image_proj(image_feat).unsqueeze(1)
        v = k

        # 文本作为 Query，图像作为 Key/Value 的跨模态注意力
        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        # 去掉序列长度维度，得到 (B, hidden_dim)
        fused = attn_output.squeeze(1)

        # 前馈网络进一步变换
        fused = self.ffn(fused)
        return fused
