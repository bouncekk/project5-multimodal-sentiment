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


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def build_vocab_from_file(train_file: str, max_samples: int = 100000) -> Vocab:
    texts = []
    with open(train_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            text = parts[1]
            texts.append(text)
    vocab = Vocab()
    vocab.build_from_texts(texts)
    return vocab


def build_model(vocab: Vocab, args: argparse.Namespace) -> nn.Module:
    text_encoder = TextEncoder(vocab_size=vocab.size, embed_dim=args.text_embed_dim, hidden_dim=args.text_hidden_dim)
    image_encoder = ImageEncoder(backbone="resnet18", pretrained=False, train_backbone=True)

    fusion = FusionModule(text_dim=text_encoder.output_dim, image_dim=image_encoder.output_dim, hidden_dim=args.fusion_hidden_dim)
    classifier = Classifier(input_dim=fusion.output_dim, num_classes=3, hidden_dim=args.cls_hidden_dim)

    class MultiModalModel(nn.Module):
        def __init__(self, text_encoder, image_encoder, fusion, classifier):
            super().__init__()
            self.text_encoder = text_encoder
            self.image_encoder = image_encoder
            self.fusion = fusion
            self.classifier = classifier

        def forward(self, input_ids, lengths, images, modality: str = "both"):
            if modality == "text_only":
                text_feat = self.text_encoder(input_ids, lengths)
                logits = self.classifier(text_feat)
                return logits
            elif modality == "image_only":
                img_feat = self.image_encoder(images)
                logits = self.classifier(img_feat)
                return logits
            else:
                text_feat = self.text_encoder(input_ids, lengths)
                img_feat = self.image_encoder(images)
                fused = self.fusion(text_feat, img_feat)
                logits = self.classifier(fused)
                return logits

    model = MultiModalModel(text_encoder, image_encoder, fusion, classifier)
    return model


def get_data_loaders(args: argparse.Namespace, vocab: Vocab) -> Tuple[DataLoader, DataLoader]:
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
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
    parser.add_argument("--text_hidden_dim", type=int, default=256)
    parser.add_argument("--fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--cls_hidden_dim", type=int, default=128)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="ratio of validation set from training data")

    parser.add_argument("--modality", type=str, default="both", choices=["both", "text_only", "image_only"], help="for ablation experiments")
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
