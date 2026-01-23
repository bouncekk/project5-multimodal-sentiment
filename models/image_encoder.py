import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """图像编码器，使用一个较小的 CNN 骨干网络（默认 ResNet18）。

    为了在作业环境中更快地训练，这里默认使用 pretrained=False，
    如果网络环境允许、训练时间充足，可以将 pretrained=True 使用预训练权重。
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = False, train_backbone: bool = True):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(pretrained=pretrained)
            out_dim = base.fc.in_features
            modules = list(base.children())[:-1]  # remove classifier
            self.backbone = nn.Sequential(*modules)
            self._output_dim = out_dim
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """参数:
        - images: (B, 3, H, W)，输入图像张量

        返回:
        - features: (B, output_dim)，每张图像的一条特征向量
        """
        x = self.backbone(images)  # (B, C, 1, 1)
        x = torch.flatten(x, 1)
        return x
