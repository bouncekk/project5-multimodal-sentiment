import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


class Vocab:
    """非常简单的词表，用于将文本分词并映射到 id。

    这里只是一个占位实现。在真实项目中，你通常会使用预训练分词器
    （例如 BERT tokenizer）。这里为了简化，只按空格切分。
    """

    def __init__(self, specials: Optional[List[str]] = None):
        if specials is None:
            specials = ["<pad>", "<unk>"]
        self.token_to_id = {}
        self.id_to_token = []
        for sp in specials:
            self.add_token(sp)
        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]

    def add_token(self, tok: str) -> int:
        if tok not in self.token_to_id:
            idx = len(self.id_to_token)
            self.token_to_id[tok] = idx
            self.id_to_token.append(tok)
        return self.token_to_id[tok]

    def build_from_texts(self, texts: List[str], min_freq: int = 1):
        from collections import Counter

        counter = Counter()
        for t in texts:
            tokens = t.strip().split()
            counter.update(tokens)
        for tok, freq in counter.items():
            if freq >= min_freq:
                self.add_token(tok)

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, max_len: int) -> Tuple[torch.Tensor, int]:
        tokens = text.strip().split()
        ids = []
        for tok in tokens[:max_len]:
            ids.append(self.token_to_id.get(tok, self.unk_id))
        length = len(ids)
        if length < max_len:
            ids.extend([self.pad_id] * (max_len - length))
        return torch.tensor(ids, dtype=torch.long), length


class MultimodalSentimentDataset(Dataset):
    """多模态（文本 + 图像）情感分类数据集。

    期望的 `train.txt` 格式（CSV，逗号分隔）:
        guid,label
    其中 label 取值为 {positive, neutral, negative} 之一，对应 data 目录下的:
        data/{guid}.jpg   图像文件
        data/{guid}.txt   文本文件

    对于 `test_without_label.txt`（没有标签）:
        guid,null
    """

    LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    def __init__(
        self,
        data_file: str,
        root_dir: str,
        vocab: Vocab,
        max_text_len: int = 64,
        is_test: bool = False,
        image_transform=None,
    ):
        self.samples = []
        self.root_dir = root_dir
        self.vocab = vocab
        self.max_text_len = max_text_len
        self.is_test = is_test
        self.image_transform = image_transform

        with open(data_file, "r", encoding="utf-8") as f:
            first = True
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 跳过首行表头 "guid,tag"
                if first and ("," in line) and ("guid" in line):
                    first = False
                    continue
                first = False

                parts = line.split(",")
                if len(parts) < 2:
                    continue

                guid = parts[0]
                # 构造图像与文本的相对路径，均位于 root_dir/data/ 下
                img_path = os.path.join("data", f"{guid}.jpg")
                text_path = os.path.join(self.root_dir, "data", f"{guid}.txt")

                # 读取对应的文本内容
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as tf:
                        text = tf.read().strip()
                else:
                    text = ""

                if is_test:
                    # 测试集没有真实标签
                    self.samples.append((img_path, text, None))
                else:
                    label_str = parts[1]
                    if label_str not in self.LABEL2ID:
                        continue
                    label_id = self.LABEL2ID[label_str]
                    self.samples.append((img_path, text, label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text, label = self.samples[idx]
        full_img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        input_ids, length = self.vocab.encode(text, self.max_text_len)

        if label is None:
            label_tensor = torch.tensor(-1, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "length": torch.tensor(length, dtype=torch.long),
            "image": image,
            "label": label_tensor,
            "raw_text": text,
            "image_path": img_path,
        }


def collate_fn(batch):
    # 在 encode() 中已经按照 max_text_len 做好 padding，这里直接 stack 即可。
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    lengths = torch.stack([item["length"] for item in batch], dim=0)
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    return {
        "input_ids": input_ids,
        "lengths": lengths,
        "images": images,
        "labels": labels,
        "raw_texts": [item["raw_text"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
    }
