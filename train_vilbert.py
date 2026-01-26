import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizerFast, BertModel, ViTModel


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ViLBERTDataset(Dataset):
    """专用于 ViLBERT 风格模型的多模态情感数据集。

    读取 train.txt / test_without_label.txt，格式：guid,label 或 guid,null。
    根据 guid 从 data/{guid}.jpg 和 data/{guid}.txt 加载图像和原始文本。
    文本编码交给 BERT tokenizer，图像预处理交给 torchvision.transforms。
    """

    def __init__(
        self,
        data_file: str,
        root_dir: str,
        is_test: bool = False,
        image_transform=None,
    ):
        self.samples = []
        self.root_dir = root_dir
        self.is_test = is_test
        self.image_transform = image_transform

        with open(data_file, "r", encoding="utf-8") as f:
            first = True
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if first and ("," in line) and ("guid" in line):
                    first = False
                    continue
                first = False

                parts = line.split(",")
                if len(parts) < 2:
                    continue

                guid = parts[0]
                img_path = os.path.join("data", f"{guid}.jpg")
                text_path = os.path.join(self.root_dir, "data", f"{guid}.txt")

                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8", errors="ignore") as tf:
                        text = tf.read().strip()
                else:
                    text = ""

                if is_test:
                    self.samples.append((guid, img_path, text, None))
                else:
                    label_str = parts[1]
                    if label_str not in LABEL2ID:
                        continue
                    label_id = LABEL2ID[label_str]
                    self.samples.append((guid, img_path, text, label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        guid, img_path, text, label = self.samples[idx]
        full_img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        if label is None:
            label_tensor = torch.tensor(-1, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "guid": guid,
            "image": image,
            "text": text,
            "label": label_tensor,
        }


def vilbert_collate_fn(batch, tokenizer, max_text_len: int):
    """将一批样本打包：BERT tokenizer 编码文本，图像堆叠为张量。"""

    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    guids = [item["guid"] for item in batch]

    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_text_len,
        return_tensors="pt",
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "images": images,
        "labels": labels,
        "guids": guids,
    }


class ViLBERTLikeModel(nn.Module):
    """简化版 ViLBERT 风格多模态模型。

    - 文本编码：BERT CLS 向量 (batch, d)
    - 图像编码：ViT pooled 输出 (batch, d_img)，投影到 d
    - 融合：将 [text_cls, image_feat] 拼接为序列长度 2，经 TransformerEncoder 融合
    - 分类：对融合后第一个 token (对应 text) 做三分类

    注意：这是一个简化版的 ViLBERT 风格 cross-modal Transformer，而不是官方权重。
    """

    def __init__(
        self,
        bert_name: str = "bert-base-uncased",
        vit_name: str = "google/vit-base-patch16-224-in21k",
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        cls_hidden_dim: int = 512,
        num_labels: int = 3,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_name)
        self.vit = ViTModel.from_pretrained(vit_name)

        # 将 ViT 的 hidden_size 投影到与 BERT 相同维度
        img_hidden = self.vit.config.hidden_size
        self.img_proj = nn.Linear(img_hidden, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask, images):
        # 文本编码
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_outputs.last_hidden_state[:, 0]  # (B, hidden_dim)

        # 图像编码
        img_outputs = self.vit(images)
        img_feat = img_outputs.pooler_output  # (B, img_hidden)
        img_feat = self.img_proj(img_feat)    # (B, hidden_dim)

        # 构造长度为 2 的序列：[text_cls, img_feat]
        fused_seq = torch.stack([text_cls, img_feat], dim=1)  # (B, 2, hidden_dim)
        fused_out = self.fusion_encoder(fused_seq)            # (B, 2, hidden_dim)

        fused_cls = fused_out[:, 0]  # 取第一个 token 作为融合后的表示
        logits = self.classifier(fused_cls)
        return logits


def get_train_val_loaders(args: argparse.Namespace, tokenizer: BertTokenizerFast) -> Tuple[DataLoader, DataLoader]:
    image_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data_path = os.path.join(args.data_dir, "train.txt")
    dataset = ViLBERTDataset(
        data_file=train_data_path,
        root_dir=args.data_dir,
        is_test=False,
        image_transform=image_transform,
    )

    val_ratio = args.val_ratio
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: vilbert_collate_fn(b, tokenizer, args.max_text_len),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: vilbert_collate_fn(b, tokenizer, args.max_text_len),
    )

    return train_loader, val_loader


def get_test_loader(args: argparse.Namespace, tokenizer: BertTokenizerFast) -> DataLoader:
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_data_path = os.path.join(args.data_dir, "test_without_label.txt")
    dataset = ViLBERTDataset(
        data_file=test_data_path,
        root_dir=args.data_dir,
        is_test=True,
        image_transform=image_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: vilbert_collate_fn(b, tokenizer, args.max_text_len),
    )

    return loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def infer(model, loader, device, output_path: str):
    model.eval()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad(), open(output_path, "w", encoding="utf-8") as fw:
        fw.write("guid,label\n")
        for batch in tqdm(loader, desc="Infer"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)
            guids = batch["guids"]

            logits = model(input_ids, attention_mask, images)
            preds = logits.argmax(dim=-1).cpu().tolist()

            for guid, pred_id in zip(guids, preds):
                label_str = ID2LABEL.get(pred_id, "neutral")
                fw.write(f"{guid},{label_str}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Training ViLBERT-like multimodal sentiment model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.txt, test_without_label.txt and data/")
    parser.add_argument("--save_dir", type=str, default="checkpoints_vilbert", help="Directory to save checkpoints")

    parser.add_argument("--max_text_len", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="ratio of validation set from training data")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bert_name", type=str, default="bert-base-uncased")
    parser.add_argument("--vit_name", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--fusion_layers", type=int, default=2)
    parser.add_argument("--fusion_heads", type=int, default=8)
    parser.add_argument("--cls_hidden_dim", type=int, default=512)

    # 推理相关参数
    parser.add_argument("--do_infer", action="store_true", help="Run inference on test_without_label.txt instead of training")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a saved ViLBERTLikeModel checkpoint for inference")
    parser.add_argument("--output_path", type=str, default="submission_vilbert.txt", help="Output file for test predictions (guid,label)")

    args = parser.parse_args()
    return args


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_name)

    # 推理模式
    if args.do_infer:
        if args.ckpt_path is None or not os.path.exists(args.ckpt_path):
            raise ValueError(f"--do_infer 需要提供有效的 --ckpt_path，当前: {args.ckpt_path}")

        print(f"[Infer] Loading checkpoint from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=device)

        model = ViLBERTLikeModel(
            bert_name=args.bert_name,
            vit_name=args.vit_name,
            hidden_dim=args.hidden_dim,
            num_layers=args.fusion_layers,
            num_heads=args.fusion_heads,
            cls_hidden_dim=args.cls_hidden_dim,
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        test_loader = get_test_loader(args, tokenizer)
        infer(model, test_loader, device, args.output_path)
        print(f"[Infer] Saved predictions to {args.output_path}")
        return

    # 训练模式
    train_loader, val_loader = get_train_val_loaders(args, tokenizer)

    model = ViLBERTLikeModel(
        bert_name=args.bert_name,
        vit_name=args.vit_name,
        hidden_dim=args.hidden_dim,
        num_layers=args.fusion_layers,
        num_heads=args.fusion_heads,
        cls_hidden_dim=args.cls_hidden_dim,
    ).to(device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("ViLBERT-like model configuration:")
    print(f"  BERT backbone: {args.bert_name}")
    print(f"  ViT backbone : {args.vit_name}")
    print(f"  Fusion layers: {args.fusion_layers}, heads: {args.fusion_heads}")
    print(f"  Total parameters: {total_params / 1e6:.3f} M, trainable: {total_trainable / 1e6:.3f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, "best_vilbert.pt")
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")


if __name__ == "__main__":
    main()
