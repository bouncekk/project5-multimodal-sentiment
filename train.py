import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from datasets.dataset import Vocab, MultimodalSentimentDataset, collate_fn
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion import FusionModule
from models.classifier import Classifier
from models.clip_encoder import CLIPTextEncoder, CLIPImageEncoder


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def build_vocab_from_file(train_file: str, max_samples: int = 100000) -> Vocab:
    """根据 CSV 格式的 train.txt (guid,tag) 构建词表。
    这里会使用 guid 到 data/{guid}.txt 中读取文本内容来统计词频。
    """
    texts = []
    with open(train_file, "r", encoding="utf-8") as f:
        first = True
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            # 跳过表头 guid,tag
            if first and ("," in line) and ("guid" in line):
                first = False
                continue
            first = False

            parts = line.split(",")
            if len(parts) < 1:
                continue
            guid = parts[0]
            text_path = os.path.join(os.path.dirname(train_file), "data", f"{guid}.txt")
            if not os.path.exists(text_path):
                continue
            # 文本文件可能不是严格的 UTF-8 编码，这里忽略无法解码的字符，避免报错中断训练
            with open(text_path, "r", encoding="utf-8", errors="ignore") as tf:
                text = tf.read().strip()
            if text:
                texts.append(text)

    vocab = Vocab()
    if texts:
        vocab.build_from_texts(texts)
    return vocab


def build_model(vocab: Vocab, args: argparse.Namespace) -> nn.Module:
    """根据 model_type 和 fusion_type 构建模型结构。
    """

    proj_dim = args.fusion_hidden_dim

    if args.model_type == "clip":
        text_encoder = CLIPTextEncoder(
            vocab_size=vocab.size,
            text_embed_dim=args.text_embed_dim,
            text_hidden_dim=args.text_hidden_dim,
            proj_dim=proj_dim,
        )
        image_encoder = CLIPImageEncoder(
            proj_dim=proj_dim,
            pretrained=True,
            train_backbone=True,
        )
    else:
        text_encoder = TextEncoder(
            vocab_size=vocab.size,
            embed_dim=args.text_embed_dim,
            hidden_dim=args.text_hidden_dim,
            num_layers=args.text_num_layers,
        )
        image_encoder = ImageEncoder(model_name="google/vit-base-patch16-224-in21k", pretrained=True, train_backbone=True)

    fusion = FusionModule(text_dim=text_encoder.output_dim, image_dim=image_encoder.output_dim, hidden_dim=args.fusion_hidden_dim)

    if args.fusion_type == "late":
        classifier_text = Classifier(input_dim=text_encoder.output_dim, num_classes=3, hidden_dim=args.cls_hidden_dim)
        classifier_image = Classifier(input_dim=image_encoder.output_dim, num_classes=3, hidden_dim=args.cls_hidden_dim)
        main_classifier = None
    elif args.fusion_type == "clip_match":
        # CLIP 匹配特征：使用余弦相似度等构造匹配特征，再送入统一分类头
        # 匹配特征维度：4 * d + 1 （t_hat, v_hat, |t_hat-v_hat|, t_hat*v_hat, cos）
        d = text_encoder.output_dim
        classifier_input_dim = 4 * d + 1
        main_classifier = Classifier(input_dim=classifier_input_dim, num_classes=3, hidden_dim=args.cls_hidden_dim)
        classifier_text = None
        classifier_image = None
    else:
        if args.modality == "text_only":
            classifier_input_dim = text_encoder.output_dim
        elif args.modality == "image_only":
            classifier_input_dim = image_encoder.output_dim
        else:  
            if args.fusion_type == "early":
                classifier_input_dim = text_encoder.output_dim + image_encoder.output_dim
            else:  
                classifier_input_dim = fusion.output_dim

        main_classifier = Classifier(input_dim=classifier_input_dim, num_classes=3, hidden_dim=args.cls_hidden_dim)
        classifier_text = None
        classifier_image = None

    class MultiModalModel(nn.Module):
        def __init__(self, text_encoder, image_encoder, fusion, main_classifier, classifier_text, classifier_image, fusion_type: str, model_type: str):
            super().__init__()
            self.text_encoder = text_encoder
            self.image_encoder = image_encoder
            self.fusion = fusion
            self.main_classifier = main_classifier
            self.classifier_text = classifier_text
            self.classifier_image = classifier_image
            self.fusion_type = fusion_type
            self.model_type = model_type

        def forward(self, input_ids, lengths, images, modality: str = "both"):
            if modality == "text_only":
                text_feat = self.text_encoder(input_ids, lengths)
                if self.fusion_type == "late" and self.classifier_text is not None:
                    logits = self.classifier_text(text_feat)
                else:
                    logits = self.main_classifier(text_feat)
                return logits
            elif modality == "image_only":
                img_feat = self.image_encoder(images)
                if self.fusion_type == "late" and self.classifier_image is not None:
                    logits = self.classifier_image(img_feat)
                else:
                    logits = self.main_classifier(img_feat)
                return logits
            else:
                text_feat = self.text_encoder(input_ids, lengths)
                img_feat = self.image_encoder(images)
                if self.fusion_type == "cross_attn":
                    fused = self.fusion(text_feat, img_feat)
                    logits = self.main_classifier(fused)
                elif self.fusion_type == "early":
                    fused = torch.cat([text_feat, img_feat], dim=-1)
                    logits = self.main_classifier(fused)
                elif self.fusion_type == "clip_match":
                    t_hat = torch.nn.functional.normalize(text_feat, dim=-1)
                    v_hat = torch.nn.functional.normalize(img_feat, dim=-1)
                    cos = (t_hat * v_hat).sum(dim=-1, keepdim=True)
                    diff = torch.abs(t_hat - v_hat)
                    prod = t_hat * v_hat
                    fused = torch.cat([t_hat, v_hat, diff, prod, cos], dim=-1)
                    logits = self.main_classifier(fused)
                else:  
                    logits_text = self.classifier_text(text_feat)
                    logits_image = self.classifier_image(img_feat)
                    logits = (logits_text + logits_image) / 2.0
                return logits

    model = MultiModalModel(text_encoder, image_encoder, fusion, main_classifier, classifier_text, classifier_image, args.fusion_type, args.model_type)
    return model


def get_data_loaders(args: argparse.Namespace, vocab: Vocab) -> Tuple[DataLoader, DataLoader]:
    image_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并缩放到 224x224
        transforms.RandomHorizontalFlip(p=0.5),               # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data_path = os.path.join(args.data_dir, "train.txt")
    dataset = MultimodalSentimentDataset(
        data_file=train_data_path,
        root_dir=args.data_dir,
        vocab=vocab,
        max_text_len=args.max_text_len,
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
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, modality: str = "both"):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths, images, modality=modality)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device, modality: str = "both"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, lengths, images, modality=modality)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Classification Training")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.txt and images")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    parser.add_argument("--max_text_len", type=int, default=64)
    parser.add_argument("--text_embed_dim", type=int, default=128)
    # 提高文本编码器容量：更大的 hidden_dim 和更多层数
    parser.add_argument("--text_hidden_dim", type=int, default=512)
    parser.add_argument("--text_num_layers", type=int, default=8)
    # 增大融合和分类头的隐藏维度，使整体参数量提升到接近 1.5e8
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--cls_hidden_dim", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="ratio of validation set from training data")

    parser.add_argument("--modality", type=str, default="both", choices=["both", "text_only", "image_only"], help="for ablation experiments")
    parser.add_argument("--fusion_type", type=str, default="cross_attn",
                        choices=["cross_attn", "early", "late", "clip_match"],
                        help="fusion strategy when using both modalities")
    parser.add_argument("--model_type", type=str, default="bert_vit",
                        choices=["bert_vit", "clip"],
                        help="backbone choice: BERT+ViT or CLIP-style encoders")
    parser.add_argument("--seed", type=int, default=42)

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

    train_file = os.path.join(args.data_dir, "train.txt")
    vocab = build_vocab_from_file(train_file)

    train_loader, val_loader = get_data_loaders(args, vocab)
    model = build_model(vocab, args).to(device)

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {args.model_type}, fusion_type: {args.fusion_type}, modality: {args.modality}")
    print(f"Total parameters: {num_params / 1e6:.3f} M, trainable: {num_trainable / 1e6:.3f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, modality=args.modality)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, modality=args.modality)

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, f"best_{args.modality}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab.token_to_id,
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")


if __name__ == "__main__":
    main()
