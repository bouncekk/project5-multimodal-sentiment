import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ClipOpenAIDataset(Dataset):
    """专用于 OpenAI CLIP 的多模态情感数据集。

    与原项目的 CSV 格式保持一致：
        guid,label

    对于每一条样本，读取：
        data/{guid}.jpg   图像文件
        data/{guid}.txt   文本文件（原始字符串）

    不再依赖 vocab.encode，文本编码交给 CLIPProcessor 完成。
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
                # 跳过首行表头 "guid,label"
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

                # 读取对应的文本内容，忽略无法用 UTF-8 解码的字符
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8", errors="ignore") as tf:
                        text = tf.read().strip()
                else:
                    text = ""

                if is_test:
                    self.samples.append((img_path, text, None))
                else:
                    label_str = parts[1]
                    if label_str not in LABEL2ID:
                        continue
                    label_id = LABEL2ID[label_str]
                    self.samples.append((img_path, text, label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text, label = self.samples[idx]
        full_img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_img_path).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        if label is None:
            label_tensor = torch.tensor(-1, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label, dtype=torch.long)

        # 关键：返回原始文本字符串，由 CLIPProcessor 负责编码
        return {
            "images": image,
            "labels": label_tensor,
            "raw_texts": text,
            "image_paths": img_path,
        }


def clip_collate_fn(batch):
    """自定义 collate_fn：

    - 保持 images 为 PIL.Image 的列表，直接交给 CLIPProcessor 处理；
    - 将 labels 堆叠为一个 LongTensor；
    - 将 raw_texts 组成字符串列表。
    """

    images = [item["images"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    texts = [item["raw_texts"] for item in batch]
    image_paths = [item["image_paths"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "raw_texts": texts,
        "image_paths": image_paths,
    }


class CLIPOpenAIWrapper(nn.Module):
    """使用 OpenAI 预训练 CLIP 模型的封装，用于情感三分类。

    - 文本 & 图像通过 CLIPProcessor 预处理
    - CLIPModel 输出 text_embeds 和 image_embeds
    - 使用归一化后的特征构造 clip_match 风格匹配向量
    - 上接一个 MLP 分类头输出 3 类情感
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", cls_hidden_dim: int = 512):
        super().__init__()
        self.model_name = model_name

        # 优先尝试离线加载（只用本地缓存，不访问网络），避免在网络被拦截时长时间重试
        try:
            self.clip = CLIPModel.from_pretrained(model_name, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
            print(f"[CLIPOpenAIWrapper] Loaded '{model_name}' from local cache (offline mode).")
        except Exception as e:
            # 如果本地没有缓存，则给出更明确的提示
            raise RuntimeError(
                f"未能在本地缓存中找到模型 '{model_name}'。\n"
                f"请确认你已经在这台机器上成功下载过该模型，或者在网络可用的环境下提前运行一次：\n"
                f"    from transformers import CLIPModel, CLIPProcessor\n"
                f"    CLIPModel.from_pretrained('{model_name}')\n"
                f"    CLIPProcessor.from_pretrained('{model_name}')\n"
                f"然后重新运行本脚本。原始错误: {e}"
            )

        # CLIP 默认投影后的文本/图像特征维度
        embed_dim = self.clip.config.projection_dim
        self.embed_dim = embed_dim

        # 匹配特征维度：t_hat, v_hat, |t-v|, t*v, cos
        match_dim = 4 * embed_dim + 1

        self.classifier = nn.Sequential(
            nn.Linear(match_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim, 3),
        )

    def forward(self, texts, images):
        """输入原始文本列表和图像张量 (B, 3, H, W)。"""
        device = next(self.parameters()).device

        # CLIPProcessor 可以直接吃 text list + PIL.Image 列表或张量。
        # 为避免 "sequence length > max_position_embeddings" 的报错，这里显式截断文本到模型允许的最大长度。
        max_txt_len = self.clip.config.text_config.max_position_embeddings

        inputs = self.processor(
            text=list(texts),
            images=images,
            return_tensors="pt",
            padding="max_length",      # 对齐到统一长度
            truncation=True,            # 超过 max_length 的部分直接截断
            max_length=max_txt_len,
        ).to(device)

        outputs = self.clip(**inputs)
        text_embeds = outputs.text_embeds  # (B, D)
        image_embeds = outputs.image_embeds  # (B, D)

        t_hat = nn.functional.normalize(text_embeds, dim=-1)
        v_hat = nn.functional.normalize(image_embeds, dim=-1)
        cos = (t_hat * v_hat).sum(dim=-1, keepdim=True)
        diff = torch.abs(t_hat - v_hat)
        prod = t_hat * v_hat
        match_feat = torch.cat([t_hat, v_hat, diff, prod, cos], dim=-1)

        logits = self.classifier(match_feat)
        return logits


def get_data_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """构造专用于 OpenAI CLIP 的 DataLoader。

    使用内部的 ClipOpenAIDataset，不再依赖 vocab.encode。
    """

    # CLIPProcessor 自己会做 resize / center crop / normalize，这里只需基础变换
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 保险起见先缩放到 224
        transforms.ToTensor(),
    ])

    train_data_path = os.path.join(args.data_dir, "train.txt")
    dataset = ClipOpenAIDataset(
        data_file=train_data_path,
        root_dir=args.data_dir,
        is_test=False,
        image_transform=image_transform,
    )

    val_ratio = args.val_ratio
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 默认的 collate_fn 即可：
    #  - 这里改为使用 clip_collate_fn，保持 images 为 PIL.Image 列表，labels 为张量
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=clip_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=clip_collate_fn,
    )

    return train_loader, val_loader


def get_test_loader(args: argparse.Namespace) -> DataLoader:
    """构造用于无标签测试集 (test_without_label.txt) 的 DataLoader。"""

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_data_path = os.path.join(args.data_dir, "test_without_label.txt")
    dataset = ClipOpenAIDataset(
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
        collate_fn=clip_collate_fn,
    )

    return loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train"):
        # 我们自定义的 Dataset 已经返回 images(list[PIL]) / labels(tensor) / raw_texts
        images = batch["images"]          # list of PIL.Image.Image, 由 CLIPProcessor 处理并移动到 device
        labels = batch["labels"].to(device)
        texts = batch["raw_texts"]

        optimizer.zero_grad()
        logits = model(texts, images)
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
            images = batch["images"]          # list of PIL.Image.Image
            labels = batch["labels"].to(device)
            texts = batch["raw_texts"]

            logits = model(texts, images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Training with OpenAI CLIP for multimodal sentiment classification")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train.txt and images")
    parser.add_argument("--save_dir", type=str, default="checkpoints_clip_openai", help="Directory to save checkpoints")

    parser.add_argument("--max_text_len", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="ratio of validation set from training data")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--cls_hidden_dim", type=int, default=512)

    # 推理相关参数（仅在 --do_infer 时使用）
    parser.add_argument("--do_infer", action="store_true", help="Run inference on test_without_label.txt instead of training")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a saved CLIPOpenAIWrapper checkpoint for inference")
    parser.add_argument("--output_path", type=str, default="submission_clip_openai.txt", help="Output file for test predictions (guid,label)")

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

    # 推理模式：只加载 checkpoint，在 test_without_label.txt 上生成预测文件
    if args.do_infer:
        if args.ckpt_path is None or not os.path.exists(args.ckpt_path):
            raise ValueError(f"--do_infer 需要提供有效的 --ckpt_path，当前: {args.ckpt_path}")

        print(f"[Infer] Loading checkpoint from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=device)

        model = CLIPOpenAIWrapper(model_name=args.clip_model_name, cls_hidden_dim=args.cls_hidden_dim).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        test_loader = get_test_loader(args)

        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with torch.no_grad(), open(args.output_path, "w", encoding="utf-8") as fw:
            fw.write("guid,label\n")
            for batch in tqdm(test_loader, desc="Infer"):
                images = batch["images"]
                texts = batch["raw_texts"]
                image_paths = batch["image_paths"]

                logits = model(texts, images)
                preds = logits.argmax(dim=-1).cpu().tolist()

                for guid_path, pred_id in zip(image_paths, preds):
                    # guid 形如 "data/xxx.jpg"，取出中间的文件名部分去掉扩展名
                    guid = os.path.splitext(os.path.basename(guid_path))[0]
                    label_str = ID2LABEL.get(pred_id, "neutral")
                    fw.write(f"{guid},{label_str}\n")

        print(f"[Infer] Saved predictions to {args.output_path}")
        return

    # 训练模式（默认）：与原先逻辑保持不变
    train_loader, val_loader = get_data_loaders(args)

    model = CLIPOpenAIWrapper(model_name=args.clip_model_name, cls_hidden_dim=args.cls_hidden_dim).to(device)

    # 打印模型参数量：总量 + 主干 CLIP + 分类头，便于与其他模型公平对比
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    clip_params = sum(p.numel() for p in model.clip.parameters())
    clip_trainable = sum(p.numel() for p in model.clip.parameters() if p.requires_grad)

    head_params = sum(p.numel() for p in model.classifier.parameters())
    head_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)

    print(f"OpenAI CLIP backbone (model_name): {args.clip_model_name}")
    print(f"  - CLIP backbone params: {clip_params / 1e6:.3f} M, trainable: {clip_trainable / 1e6:.3f} M")
    print(f"  - Classification head params: {head_params / 1e6:.3f} M, trainable: {head_trainable / 1e6:.3f} M")
    print(f"  - Total parameters (CLIP + head): {total_params / 1e6:.3f} M, trainable: {total_trainable / 1e6:.3f} M")

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
            ckpt_path = os.path.join(args.save_dir, "best_clip_openai.pt")
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")


if __name__ == "__main__":
    main()
