import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class TextEncoder(nn.Module):
    """基于 BERT 结构的文本编码器。
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()

        # 为当前任务构建一个 BERT 配置
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=max(1, hidden_dim // 64),  
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

        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        range_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        lengths_expanded = lengths.unsqueeze(1).expand(-1, seq_len)
        attention_mask = (range_ids < lengths_expanded).long()

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.pooler_output is not None:
            features = outputs.pooler_output  # (B, hidden_size)
        else:
            features = outputs.last_hidden_state[:, 0, :]
        return features
