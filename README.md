# Project5: 多模态情感分类

任务：给定配对的 **文本 + 图像**，预测三分类情感标签：`negative`、`neutral`、`positive`。

本仓库提供了一个**可复现的基线模型（baseline）**，主要特点包括：
- **多模态融合**（text + image）
- **Text-only / Image-only 消融实验**（满足作业对消融实验的要求）
- **在训练集中自动划分验证集 + 便于调参的接口**
- **训练 / 验证 / 推理脚本**（train / eval / infer）

## 目录结构

```text
project5/
├── data/
│   ├── train.txt
│   └── test_without_label.txt
├── models/
│   ├── text_encoder.py
│   ├── image_encoder.py
│   ├── fusion.py
│   └── classifier.py
├── datasets/
│   └── dataset.py
├── train.py
├── eval.py
├── infer.py
├── requirements.txt
└── README.md
```

## 数据格式

- `data/train.txt` (tab-separated):

```text
image_path\ttext\tlabel
```

含义说明：
- `image_path`：相对于 `data/` 的图像路径（例如 `images/0001.jpg`）
- `text`：文本描述
- `label`：情感标签，取值为 `negative`、`neutral`、`positive` 之一

- `data/test_without_label.txt`（制表符分隔）：

```text
image_path\ttext
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 训练（包含验证集划分与调参接口）

```bash
python train.py \
  --data_dir data \
  --save_dir checkpoints \
  --val_ratio 0.1 \
  --batch_size 16 \
  --epochs 10 \
  --lr 1e-3 \
  --modality both
```

用于**消融实验**的关键参数：
- `--modality both`      ：文本 + 图像（完整多模态模型）
- `--modality text_only` ：仅使用文本分支（text-only）
- `--modality image_only`：仅使用图像分支（image-only）

验证集划分由脚本自动完成：从 `train.txt` 中按照 `--val_ratio` 比例划分出验证集。

## 在有标签文件上评估模型

如果你有单独的开发集 / 验证集文件，例如 `data/dev.txt`：

```bash
python eval.py \
  --data_dir data \
  --data_file dev.txt \
  --ckpt_path checkpoints/best_both.pt \
  --modality both
```

## 在测试集上进行推理

对 `data/test_without_label.txt` 进行情感标签预测：

```bash
python infer.py \
  --data_dir data \
  --test_file test_without_label.txt \
  --ckpt_path checkpoints/best_both.pt \
  --output_file predictions.txt \
  --modality both
```

生成的 `predictions.txt` 中，每一行对应一条样本的预测标签（`negative`、`neutral`、`positive` 之一）。

## 模型概览

- `models/text_encoder.py` ：Embedding + 双向 LSTM 文本编码器
- `models/image_encoder.py`：ResNet18 骨干网络的图像编码器
- `models/fusion.py`       ：拼接（concatenation）+ MLP 的融合模块
- `models/classifier.py`   ：用于三分类情感任务的 MLP 分类器

你可以在保持训练 / 验证 / 推理脚本不变的前提下，自由替换上述任意模块
（例如：文本部分换成 BERT，图像部分换成更强的 CNN / Vision Transformer，
或将融合模块改为注意力、跨模态对齐等更复杂的结构）。
