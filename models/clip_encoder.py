import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder


class CLIPTextEncoder(nn.Module):
    """基于现有 TextEncoder 的 CLIP 风格文本编码器。

    内部复用当前的 TextEncoder，然后通过一个线性投影层将文本特征映射到统一的对齐维度，
    便于与图像特征进行余弦相似度计算和匹配特征构造。
    """

    def __init__(self, vocab_size: int, text_embed_dim: int, text_hidden_dim: int, proj_dim: int):
        super().__init__()
        self.base_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
        )
        self.proj = nn.Linear(self.base_encoder.output_dim, proj_dim)
        self._output_dim = proj_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        base_feat = self.base_encoder(input_ids, lengths)  # (B, D_base)
        feat = self.proj(base_feat)  # (B, proj_dim)
        return feat


class CLIPImageEncoder(nn.Module):
    """基于现有 ImageEncoder 的 CLIP 风格图像编码器。

    内部复用当前的 ViT 图像编码器，然后通过一个线性投影层将图像特征映射到统一的对齐维度。
    """

    def __init__(self, proj_dim: int, pretrained: bool = True, train_backbone: bool = True):
        super().__init__()
        self.base_encoder = ImageEncoder(
            model_name="google/vit-base-patch16-224-in21k",
            pretrained=pretrained,
            train_backbone=train_backbone,
        )
        self.proj = nn.Linear(self.base_encoder.output_dim, proj_dim)
        self._output_dim = proj_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        base_feat = self.base_encoder(images)  # (B, D_base)
        feat = self.proj(base_feat)  # (B, proj_dim)
        return feat
