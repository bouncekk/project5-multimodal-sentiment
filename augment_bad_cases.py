import os
import csv
import random
from typing import List

from PIL import Image, ImageEnhance


DATA_DIR = "."  
BADCASE_CSV = os.path.join(DATA_DIR, "val_bad_cases_epoch10.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
DATA_FOLDER = os.path.join(DATA_DIR, "data")

# 每个 bad case 生成多少个增强样本
N_TEXT_AUG_PER_SAMPLE = 1
N_IMAGE_AUG_PER_SAMPLE = 1


def load_bad_cases(path: str) -> List[tuple]:
    bad_cases = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            guid = row["guid"].strip()
            true_label = row["true_label"].strip()
            if not guid:
                continue
            bad_cases.append((guid, true_label))
    return bad_cases


def simple_text_augment(text: str) -> str:
    """非常简单的中文情感文本增强：插入/替换一些语气词，保持语义不变。
    """
    fillers = ["其实", "真的", "就是", "有点", "有点儿", "还挺", "非常", "特别"]
    # 按句号/叹号分句，随机在句子前面插入一些语气词
    sentences = [s for s in text.replace("!", "。").split("。") if s.strip()]
    if not sentences:
        return text

    new_sentences = []
    for s in sentences:
        s_strip = s.strip()
        if not s_strip:
            continue
        if random.random() < 0.5:
            filler = random.choice(fillers)
            new_sentences.append(f"{filler}{s_strip}")
        else:
            new_sentences.append(s_strip)

    return "。".join(new_sentences)


def augment_image(img: Image.Image) -> Image.Image:
    """对图像做轻量增强：随机翻转 + 亮度/对比度扰动。"""
    out = img.copy()

    # 随机水平翻转
    if random.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)

    # 轻微亮度和对比度扰动
    b_factor = random.uniform(0.9, 1.1)
    c_factor = random.uniform(0.9, 1.1)
    out = ImageEnhance.Brightness(out).enhance(b_factor)
    out = ImageEnhance.Contrast(out).enhance(c_factor)

    return out


def main():
    if not os.path.exists(BADCASE_CSV):
        raise FileNotFoundError(f"Bad case CSV not found: {BADCASE_CSV}")
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"train.txt not found: {TRAIN_FILE}")
    if not os.path.isdir(DATA_FOLDER):
        raise FileNotFoundError(f"data folder not found: {DATA_FOLDER}")

    bad_cases = load_bad_cases(BADCASE_CSV)
    print(f"Loaded {len(bad_cases)} bad cases from {BADCASE_CSV}")

    # 读取当前已有的 guid 集合，避免重复
    existing_guids = set()
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
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
            if len(parts) >= 1:
                existing_guids.add(parts[0])

    new_train_lines: List[str] = []

    for guid, true_label in bad_cases:
        txt_path = os.path.join(DATA_FOLDER, f"{guid}.txt")
        img_path = os.path.join(DATA_FOLDER, f"{guid}.jpg")

        # 文本增强
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as tf:
                original_text = tf.read()
            for i in range(N_TEXT_AUG_PER_SAMPLE):
                aug_text = simple_text_augment(original_text)
                new_guid = f"{guid}_t{i+1}"
                if new_guid in existing_guids:
                    continue
                new_txt_path = os.path.join(DATA_FOLDER, f"{new_guid}.txt")
                with open(new_txt_path, "w", encoding="utf-8") as nf:
                    nf.write(aug_text)
                # 如果原图存在，可以顺便复制一份保持 guid 对齐
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    new_img_path = os.path.join(DATA_FOLDER, f"{new_guid}.jpg")
                    img.save(new_img_path)
                new_train_lines.append(f"{new_guid},{true_label}")
                existing_guids.add(new_guid)

        # 图像增强
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            for i in range(N_IMAGE_AUG_PER_SAMPLE):
                aug_img = augment_image(img)
                new_guid = f"{guid}_i{i+1}"
                if new_guid in existing_guids:
                    continue
                new_img_path = os.path.join(DATA_FOLDER, f"{new_guid}.jpg")
                aug_img.save(new_img_path)
                # 如果有原始文本，直接复用；否则写空串
                txt_for_new = ""
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as tf:
                        txt_for_new = tf.read()
                new_txt_path = os.path.join(DATA_FOLDER, f"{new_guid}.txt")
                with open(new_txt_path, "w", encoding="utf-8") as nf:
                    nf.write(txt_for_new)
                new_train_lines.append(f"{new_guid},{true_label}")
                existing_guids.add(new_guid)

    # 追加到 train.txt
    if new_train_lines:
        with open(TRAIN_FILE, "a", encoding="utf-8") as f:
            for line in new_train_lines:
                f.write("\n" + line)
        print(f"Appended {len(new_train_lines)} augmented samples to {TRAIN_FILE}")
    else:
        print("No new augmented samples were created.")


if __name__ == "__main__":
    main()
