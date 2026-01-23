import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """简单的文本编码器：Embedding + 双向 LSTM（BiLSTM）。

    这是一个轻量级 baseline，后续可以替换成 BERT/中文 BERT 等更强的文本编码器。
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    @property
    def output_dim(self) -> int:
        return self.lstm.hidden_size * 2

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """参数:
        - input_ids: (batch, seq_len)，文本 token 的 id 序列
        - lengths: (batch,)，未 padding 之前的真实序列长度

        返回:
        - features: (batch, output_dim)，每个样本的一条文本特征向量
        """
        embedded = self.embedding(input_ids)  # (B, T, E)

        # 将带 padding 的序列打包，便于 LSTM 按真实长度计算
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, batch, hidden_dim)
        h_n = h_n.view(self.lstm.num_layers, 2, input_ids.size(0), self.lstm.hidden_size)
        # 取最后一层、两个方向的隐状态并拼接
        last_layer_h = h_n[-1]  # (2, B, H)
        features = torch.cat([last_layer_h[0], last_layer_h[1]], dim=-1)  # (B, 2H)
        return features
