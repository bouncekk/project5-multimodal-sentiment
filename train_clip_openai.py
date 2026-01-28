import os
import re
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


def clean_text(text: str) -> str:
    # 去掉 URL
    text = re.sub(r"https?://\S+", "", text)
    # 去掉 @用户名
    text = re.sub(r"@[A-Za-z0-9_\-]+", "", text)
    # 将换行替换为空格
    text = text.replace("\r", " ").replace("\n", " ")
    # 合并多余空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ClipOpenAIDataset(Dataset):
    """专用于 OpenAI CLIP 的多模态情感数据集。
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

                # 读取对应的文本内容，忽略无法用 UTF-8 解码的字符
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8", errors="ignore") as tf:
                        raw_text = tf.read()
                        text = clean_text(raw_text)
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

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", cls_hidden_dim: int = 512, fusion_type: str = "clip_match"):
        super().__init__()
        self.model_name = model_name
        self.fusion_type = fusion_type

        # 优先尝试离线加载
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

        embed_dim = self.clip.config.projection_dim
        self.embed_dim = embed_dim

        match_dim = 4 * embed_dim + 1

        self.clip_classifier = nn.Sequential(
            nn.Linear(match_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim, 3),
        )

        self.early_classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim, 3),
        )

        self.text_classifier = nn.Linear(embed_dim, 3)
        self.image_classifier = nn.Linear(embed_dim, 3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        # 稍微加深层数，加强跨模态交互能力
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 残差门控：在融合后的表示上加入一条来自文本特征 t_hat 的可学习门控残差
        self.attn_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
        )

        # 更强的分类头：LayerNorm + GELU + 两层 MLP
        self.attn_classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim, cls_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim // 2, 3),
        )

        self.text_only_classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim, 3),
        )

        self.image_only_classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(cls_hidden_dim, 3),
        )

    def forward(self, texts, images):
        """输入原始文本列表和图像张量 (B, 3, H, W)。"""
        device = next(self.parameters()).device

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

        if self.fusion_type == "clip_match":
            cos = (t_hat * v_hat).sum(dim=-1, keepdim=True)
            diff = torch.abs(t_hat - v_hat)
            prod = t_hat * v_hat
            match_feat = torch.cat([t_hat, v_hat, diff, prod, cos], dim=-1)
            logits = self.clip_classifier(match_feat)

        elif self.fusion_type == "early":
            feat = torch.cat([t_hat, v_hat], dim=-1)
            logits = self.early_classifier(feat)

        elif self.fusion_type == "late":
            logits_t = self.text_classifier(t_hat)
            logits_v = self.image_classifier(v_hat)
            logits = (logits_t + logits_v) / 2.0

        elif self.fusion_type == "attn":
            seq = torch.stack([t_hat, v_hat], dim=1)  
            fused = self.attn_encoder(seq)
            fused_cls = fused[:, 0]  

            gate = self.attn_gate(torch.cat([fused_cls, t_hat], dim=-1)) 
            fused_cls = fused_cls + gate * t_hat

            logits = self.attn_classifier(fused_cls)

        elif self.fusion_type == "text_only":
            logits = self.text_only_classifier(t_hat)

        elif self.fusion_type == "image_only":
            logits = self.image_only_classifier(v_hat)

        else:
            cos = (t_hat * v_hat).sum(dim=-1, keepdim=True)
            diff = torch.abs(t_hat - v_hat)
            prod = t_hat * v_hat
            match_feat = torch.cat([t_hat, v_hat, diff, prod, cos], dim=-1)
            logits = self.clip_classifier(match_feat)

        return logits


def get_data_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """构造专用于 OpenAI CLIP 的 DataLoader。
    """

    image_transform = transforms.Resize((224, 224))

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

    image_transform = transforms.Resize((224, 224))

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
        images = batch["images"]          
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


def evaluate(model, loader, criterion, device, log_badcase_path: str | None = None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    fw = None
    if log_badcase_path is not None:
        os.makedirs(os.path.dirname(log_badcase_path) or ".", exist_ok=True)
        fw = open(log_badcase_path, "w", encoding="utf-8")
        fw.write("guid,true_label,pred_label\n")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            images = batch["images"]         
            labels = batch["labels"].to(device)
            texts = batch["raw_texts"]

            logits = model(texts, images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if fw is not None:
                image_paths = batch.get("image_paths", None)
                if image_paths is not None:
                    preds_cpu = preds.cpu().tolist()
                    labels_cpu = labels.cpu().tolist()
                    for guid_path, y_true, y_pred in zip(image_paths, labels_cpu, preds_cpu):
                        guid = os.path.splitext(os.path.basename(guid_path))[0]
                        if y_true != y_pred:
                            fw.write(f"{guid},{ID2LABEL[int(y_true)]},{ID2LABEL[int(y_pred)]}\n")

    if fw is not None:
        fw.close()

    return total_loss / total, correct / total


def evaluate_ablation(model, loader, device, mode: str):
    """在验证集上做消融实验：
    mode:
        - "full": 正常多模态
        - "text_only": 文本单模态（图像信息被抹掉）
        - "image_only": 图像单模态（文本信息被抹掉）
    """

    assert mode in {"full", "text_only", "image_only"}

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Val[{mode}]"):
            images = batch["images"]         
            labels = batch["labels"].to(device)
            texts = batch["raw_texts"]

            if mode == "text_only":
                from PIL import Image as _Image

                dummy_images = []
                for _ in images:
                    dummy_images.append(_Image.new("RGB", (224, 224), (0, 0, 0)))
                images = dummy_images

            elif mode == "image_only":
                texts = ["" for _ in texts]

            logits = model(texts, images)
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


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

    parser.add_argument("--fusion_type", type=str, default="clip_match",
                        choices=["clip_match", "early", "late", "attn", "text_only", "image_only"],
                        help="Fusion strategy on top of OpenAI CLIP embeddings (including text_only / image_only variants)")

    parser.add_argument("--do_infer", action="store_true", help="Run inference on test_without_label.txt instead of training")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to a saved CLIPOpenAIWrapper checkpoint for inference/ablation eval")
    parser.add_argument("--do_ablation_eval", action="store_true", help="Run ablation evaluation (full/text-only/image-only) on validation set using a trained checkpoint")
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

        model = CLIPOpenAIWrapper(model_name=args.clip_model_name,
                                  cls_hidden_dim=args.cls_hidden_dim,
                                  fusion_type=args.fusion_type).to(device)
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
                    guid = os.path.splitext(os.path.basename(guid_path))[0]
                    label_str = ID2LABEL.get(pred_id, "neutral")
                    fw.write(f"{guid},{label_str}\n")

        print(f"[Infer] Saved predictions to {args.output_path}")
        return

    # 消融验证模式：使用已训练好的 checkpoint，在验证集上做 full/text-only/image-only 三种评估
    if args.do_ablation_eval:
        if args.ckpt_path is None or not os.path.exists(args.ckpt_path):
            raise ValueError(f"--do_ablation_eval 需要提供有效的 --ckpt_path，当前: {args.ckpt_path}")

        print(f"[AblationEval] Loading checkpoint from {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location=device)

        _, val_loader = get_data_loaders(args)

        model = CLIPOpenAIWrapper(
            model_name=args.clip_model_name,
            cls_hidden_dim=args.cls_hidden_dim,
            fusion_type=args.fusion_type,
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)

        acc_full = evaluate_ablation(model, val_loader, device, mode="full")
        acc_text = evaluate_ablation(model, val_loader, device, mode="text_only")
        acc_image = evaluate_ablation(model, val_loader, device, mode="image_only")

        print("[AblationEval] Validation accuracy:")
        print(f"  - full (text+image):  {acc_full * 100:.2f}%")
        print(f"  - text only:          {acc_text * 100:.2f}%")
        print(f"  - image only:         {acc_image * 100:.2f}%")

        return

    train_loader, val_loader = get_data_loaders(args)

    model = CLIPOpenAIWrapper(
        model_name=args.clip_model_name,
        cls_hidden_dim=args.cls_hidden_dim,
        fusion_type=args.fusion_type,
    ).to(device)

    # 打印模型参数量：总量 + 主干 CLIP + 分类头，便于与其他模型公平对比
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    clip_params = sum(p.numel() for p in model.clip.parameters())
    clip_trainable = sum(p.numel() for p in model.clip.parameters() if p.requires_grad)

    head_params = sum(p.numel() for p in model.clip_classifier.parameters())
    head_trainable = sum(p.numel() for p in model.clip_classifier.parameters() if p.requires_grad)

    print(f"OpenAI CLIP backbone (model_name): {args.clip_model_name}, fusion_type: {args.fusion_type}")
    print(f"  - CLIP backbone params: {clip_params / 1e6:.3f} M, trainable: {clip_trainable / 1e6:.3f} M")
    print(f"  - clip_match head params: {head_params / 1e6:.3f} M, trainable: {head_trainable / 1e6:.3f} M")
    print(f"  - Total parameters (CLIP + all heads): {total_params / 1e6:.3f} M, trainable: {total_trainable / 1e6:.3f} M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        badcase_csv = os.path.join(args.save_dir, f"val_bad_cases_epoch{epoch}.csv")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, log_badcase_path=badcase_csv)

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
