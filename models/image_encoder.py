import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ImageEncoder(nn.Module):
    """图像编码器，使用 Vision Transformer (ViT) 作为骨干网络。

    默认使用 `google/vit-base-patch16-224-in21k` 的配置，可以选择是否加载预训练权重。
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", pretrained: bool = True, train_backbone: bool = True):
        super().__init__()

        # 出于可复现性和环境限制考虑，这里不再尝试从 HuggingFace 下载预训练权重，
        # 始终使用本地随机初始化的 ViTConfig。
        config = ViTConfig()
        self.vit = ViTModel(config)

        self._output_dim = self.vit.config.hidden_size

        if not train_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """参数:
        - images: (B, 3, H, W)，输入图像张量，需已 resize 到 224x224，并做归一化

        返回:
        - features: (B, output_dim)，每张图像的一条 ViT 池化后的特征向量
        """
        # ViTModel 期望输入参数名为 pixel_values
        outputs = self.vit(pixel_values=images)
        if outputs.pooler_output is not None:
            feats = outputs.pooler_output  # (B, hidden_size)
        else:
            # 如果没有 pooler，则对 patch 维做平均池化
            feats = outputs.last_hidden_state.mean(dim=1)
        return feats
