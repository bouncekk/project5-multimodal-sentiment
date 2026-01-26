import os
import argparse

import torch
from torch.utils.data import DataLoader
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


def load_vocab_from_ckpt(ckpt_path: str) -> Vocab:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    token_to_id = ckpt["vocab"]
    vocab = Vocab()
    vocab.token_to_id = token_to_id
    vocab.id_to_token = [None] * len(token_to_id)
    for tok, idx in token_to_id.items():
        vocab.id_to_token[idx] = tok
    vocab.pad_id = vocab.token_to_id["<pad>"]
    vocab.unk_id = vocab.token_to_id["<unk>"]
    return vocab


def build_model_from_ckpt(ckpt_path: str, vocab: Vocab, modality: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args_dict = ckpt["args"]

    model_type = args_dict.get("model_type", "bert_vit")

    if model_type == "clip":
        text_encoder = CLIPTextEncoder(
            vocab_size=vocab.size,
            text_embed_dim=args_dict["text_embed_dim"],
            text_hidden_dim=args_dict["text_hidden_dim"],
            proj_dim=args_dict["fusion_hidden_dim"],
        )
        image_encoder = CLIPImageEncoder(
            proj_dim=args_dict["fusion_hidden_dim"],
            pretrained=True,
            train_backbone=True,
        )
    else:
        text_encoder = TextEncoder(vocab_size=vocab.size, embed_dim=args_dict["text_embed_dim"], hidden_dim=args_dict["text_hidden_dim"])
        # 使用 ViT 版本的图像编码器，保持与训练时一致
        image_encoder = ImageEncoder(model_name="google/vit-base-patch16-224-in21k", pretrained=True, train_backbone=True)

    fusion = FusionModule(text_dim=text_encoder.output_dim, image_dim=image_encoder.output_dim, hidden_dim=args_dict["fusion_hidden_dim"])

    fusion_type = args_dict.get("fusion_type", "cross_attn")

    if fusion_type == "late":
        classifier_text = Classifier(input_dim=text_encoder.output_dim, num_classes=3, hidden_dim=args_dict["cls_hidden_dim"])
        classifier_image = Classifier(input_dim=image_encoder.output_dim, num_classes=3, hidden_dim=args_dict["cls_hidden_dim"])
        main_classifier = None
    elif fusion_type == "clip_match":
        d = text_encoder.output_dim
        classifier_input_dim = 4 * d + 1
        main_classifier = Classifier(input_dim=classifier_input_dim, num_classes=3, hidden_dim=args_dict["cls_hidden_dim"])
        classifier_text = None
        classifier_image = None
    else:
        if modality == "text_only":
            classifier_input_dim = text_encoder.output_dim
        elif modality == "image_only":
            classifier_input_dim = image_encoder.output_dim
        else:
            if fusion_type == "early":
                classifier_input_dim = text_encoder.output_dim + image_encoder.output_dim
            else:
                classifier_input_dim = fusion.output_dim

        main_classifier = Classifier(input_dim=classifier_input_dim, num_classes=3, hidden_dim=args_dict["cls_hidden_dim"])
        classifier_text = None
        classifier_image = None

    class MultiModalModel(torch.nn.Module):
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

    model = MultiModalModel(text_encoder, image_encoder, fusion, main_classifier, classifier_text, classifier_image, fusion_type, model_type)
    model.load_state_dict(ckpt["model_state"])
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on test_without_label.txt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--test_file", type=str, default="test_without_label.txt")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="predictions.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_text_len", type=int, default=64)
    parser.add_argument("--modality", type=str, default="both", choices=["both", "text_only", "image_only"])
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = load_vocab_from_ckpt(args.ckpt_path)
    model = build_model_from_ckpt(args.ckpt_path, vocab, args.modality).to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_path = os.path.join(args.data_dir, args.test_file)
    dataset = MultimodalSentimentDataset(
        data_file=test_path,
        root_dir=args.data_dir,
        vocab=vocab,
        max_text_len=args.max_text_len,
        is_test=True,
        image_transform=image_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Infer"):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            images = batch["images"].to(device)

            logits = model(input_ids, lengths, images, modality=args.modality)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for p in all_preds:
            label_str = ID2LABEL[p]
            f.write(label_str + "\n")

    print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()
