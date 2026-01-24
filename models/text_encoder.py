import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class TextEncoder(nn.Module):
    """基于 BERT 结构的文本编码器。

    这里使用 transformers 提供的 BertModel，但词表大小由当前任务的 Vocab 决定，
    等价于一个从头训练的 BERT encoder，用于提取句子级别的表示。
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 4, dropout: float = 0.1):
        """参数保持与旧接口兼容，但内部改为 BERT 配置。

        - vocab_size: 由 Vocab.size 决定
        - hidden_dim: BERT 的 hidden_size
        - num_layers: Transformer encoder 层数
        - dropout: 隐层 dropout
        """
        super().__init__()

        # 为当前任务构建一个 BERT 配置（从头训练，而非加载预训练权重）
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=max(1, hidden_dim // 64),  # 保证可被 head 数整除
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.bert = BertModel(config)

    @property
    def output_dim(self) -> int:
        return self.bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """参数:
        - input_ids: (batch, seq_len)，文本 token 的 id 序列
        - lengths: (batch,)，未 padding 之前的真实序列长度

        返回:
        - features: (batch, output_dim)，每个样本的一条文本特征向量
        """

        # 根据真实长度构造 attention mask: padding 位置为 0，其余为 1
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        # (B, seq_len)
        range_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        lengths_expanded = lengths.unsqueeze(1).expand(-1, seq_len)
        attention_mask = (range_ids < lengths_expanded).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] 位置的向量（pooler_output）作为句子级表示
        if outputs.pooler_output is not None:
            features = outputs.pooler_output  # (B, hidden_size)
        else:
            # 部分配置可能没有 pooler，这种情况下取 CLS token 的 hidden state
            features = outputs.last_hidden_state[:, 0, :]
        return features
