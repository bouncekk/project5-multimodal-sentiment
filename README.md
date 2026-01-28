# Project5: 多模态情感分类（文本 + 图像）

任务：给定配对的 文本 + 图像，预测三分类情感标签：negative、neutral、positive。

本目录下的代码支持多种编码器与融合策略的版本，包括：

- 自实现 BERT+ViT 基线模型（从头训练的 BERT 风格文本编码器 + ViT 图像编码器 + 跨模态注意力融合）。
- 预训练 ViLBERT 风格模型（`train_vilbert.py`）。
- 预训练 OpenAI CLIP 编码器模型（`train_clip_openai.py`），支持多种融合方式：
  - `clip_match` / `early` / `late` / `attn` / `text_only` / `image_only`。
- 基于 bad case 的数据增强脚本（`augment_bad_cases.py`）。

所有实验所需的脚本均位于当前 `project5` 目录下，可在同一环境中复现报告中的主要结果。

## 目录结构

```text
project5/
├── data/                           
│   ├── <guid>.jpg                
│   └── <guid>.txt                
├── train.txt 
├── test_without_label.txt 
├── datasets/
│   └── dataset.py                 # baseline 数据集定义（BERT+ViT baseline 使用）
├── models/
│   ├── text_encoder.py            # BERT 文本编码器（从头训练）
│   ├── image_encoder.py           # ViT 图像编码器
│   ├── fusion.py                  # 融合模块
│   └── classifier.py              # 分类头
├── train.py                       # BERT+ViT训练 / 验证 / 预测
├── train_vilbert.py               # ViLBERT 预训练模型训练 / 预测
├── train_clip_openai.py           # OpenAI CLIP 模型训练 / 预测 / 消融
├── eval.py                        # baseline验证（针对 train.py）
├── infer.py                       # baseline预测（针对 train.py）
├── augment_bad_cases.py           # 基于 bad case 的数据增强脚本
├── val_bad_cases_epoch10.csv      # 一次训练得到的验证集 bad case 列表示例
├── checkpoints*/                  # 各类模型的 checkpoint 输出目录
├── requirements.txt               # 依赖列表
├── 实验报告.md                    # 实验报告
└── README.md                      # 本说明文件
```

## 环境与依赖

建议使用 Python 3.10+，配合支持 GPU 的 PyTorch。基本依赖已在 `requirements.txt` 中给出。

安装方式：

```bash
pip install -r requirements.txt
```

模型依赖的预训练权重（如 ViT、BERT、OpenAI CLIP）已通过 `transformers` 的离线缓存机制使用，示例路径如：

- ViT：`/mnt/workspace/project5-multimodal-sentiment/models--google--vit-base-patch16-224-in21k/...`
- BERT：`/mnt/workspace/project5-multimodal-sentiment/models--bert-base-uncased/...`
- CLIP：`/mnt/workspace/project5-multimodal-sentiment/models--openai--clip-vit-base-patch32/...`

在无网络环境下，需要事先将对应模型的 snapshot 放到上述路径，并在命令行中传入相应 `--bert_name` / `--vit_name` / `--clip_model_name`。


## 实验流程

本项目主要包含三类模型实验：

1. 自实现 BERT+ViT 基线（`train.py`）。  
2. 预训练 ViLBERT 风格模型（`train_vilbert.py`）。  
3. 预训练 OpenAI CLIP 编码器模型（`train_clip_openai.py`），含多种融合策略与消融实验。  

### 1. 训练 BERT+ViT 基线模型

```bash
python train.py \
  --data_dir . \
  --save_dir checkpoints_bert_vit_big \
  --val_ratio 0.1 \
  --batch_size 8 \
  --epochs 10 \
  --lr 1e-4 \
  --modality both \
  --model_type bert_vit \
  --fusion_type cross_attn \
  --text_hidden_dim 512 \
  --text_num_layers 8 \
  --fusion_hidden_dim 512 \
  --cls_hidden_dim 256
```

- 程序会从 `train.txt` 自动划分 10% 作为验证集。
- 最优模型会保存在 `checkpoints_bert_vit_big` 下（文件名可在脚本中查看）。

如需在测试集上推理，可使用 `infer.py` 或在 `train.py` 中添加相应推理逻辑（原项目已提供基础版本）。

### 2. 训练预训练 ViLBERT 风格模型

```bash
python train_vilbert.py \
  --data_dir . \
  --save_dir checkpoints_vilbert \
  --batch_size 8 \
  --epochs 10 \
  --lr 1e-4 \
  --val_ratio 0.1 \
  --bert_name "path/to/bert/snapshot" \
  --vit_name "path/to/vit/snapshot" \
  --hidden_dim 768 \
  --fusion_layers 2 \
  --fusion_heads 8 \
  --cls_hidden_dim 512
```

训练结束后会在 `checkpoints_vilbert` 下保存最优模型。

### 3. 训练 OpenAI CLIP 编码器模型（多种融合方式）

最核心的实验来自 `train_clip_openai.py`，以预训练 CLIP 为骨干，比较不同融合策略。

#### 3.1 训练多模态 CLIP 模型（示例：attn 融合）

```bash
python train_clip_openai.py \
  --data_dir . \
  --save_dir checkpoints_clip_openai \
  --batch_size 8 \
  --epochs 10 \
  --lr 1e-5 \
  --val_ratio 0.1 \
  --clip_model_name "path/to/clip/snapshot" \
  --cls_hidden_dim 512 \
  --fusion_type attn
```

- `fusion_type` 可选：`clip_match` / `early` / `late` / `attn` / `text_only` / `image_only`。
- 程序会打印 CLIP 主干与分类头的参数量，自动在 `train.txt` 中划分验证集。

#### 3.2 在测试集上推理

以 attn 融合模型为例：

```bash
python train_clip_openai.py \
  --data_dir . \
  --save_dir checkpoints_clip_openai \
  --batch_size 8 \
  --clip_model_name "path/to/clip/snapshot" \
  --cls_hidden_dim 512 \
  --fusion_type attn \
  --do_infer \
  --ckpt_path checkpoints_clip_openai/best_clip_openai.pt \
  --output_path submission_clip_openai_attn.txt
```

输出文件 `submission_clip_openai_attn.txt` 

#### 3.3 消融实验

1. 纯文本 / 纯图像 CLIP 模型（单模态基线）：
   单独训练只依赖文本或图像的 CLIP 分类头，用于与多模态模型对比：换 `--fusion_type` 为 `text_only` / `image_only`即可。

2. 使用 `--do_ablation_eval` 可以在验证集上对同一模型做多模态 / 文本单模态 / 图像单模态三种评估：

```bash
python train_clip_openai.py \
  --data_dir . \
  --save_dir checkpoints_clip_openai \
  --batch_size 8 \
  --val_ratio 0.1 \
  --clip_model_name "path/to/clip/snapshot" \
  --cls_hidden_dim 512 \
  --fusion_type attn \
  --do_ablation_eval \
  --ckpt_path checkpoints_clip_openai/best_clip_openai.pt
```

脚本会打印：

- full (text+image) 的验证集准确率。
- text only / image only 的验证集准确率。


### 4. 基于 bad case 的数据增强 + 重新训练

`train_clip_openai.py` 在验证阶段可以自动记录 bad case 到 CSV（例如 `val_bad_cases_epoch10.csv`），然后可以利用 `augment_bad_cases.py` 做针对性数据增强：

```bash
python augment_bad_cases.py
```

脚本会：
读取 bad case文件，对对应的 data 做轻量增强（文本插入语气词、图像翻转+亮度/对比度扰动）。   
生成新的 guid（如 `guid_t1`, `guid_i1`），写入 `data/` 并在 `train.txt` 末尾追加新样本行。

然后可以在增强后的数据上重新训练 CLIP 模型，例如：

```bash
python train_clip_openai.py --data_dir . --save_dir checkpoints_clip_openai_attn_clean --batch_size 8 --epochs 10 --lr 1e-5 --val_ratio 0.1 --clip_model_name "path/to/clip/snapshot" --cls_hidden_dim 512 --fusion_type attn
```

通过比较增强前后的验证集准确率与 bad case 分布，可以分析数据层面的改进效果。

## 模型超参数

为保证不同模型之间的公平对比，除非特别说明，所有实验统一采用如下训练设置：

- batch size：8  
- epochs：10  
- 优化器：AdamW  
- learning rate：1e-5
- weight_decay：1e-4（仅 OpenAI CLIP）  
- 验证集划分：`val_ratio=0.1`（从训练集随机划分 10% 作为验证集）  
- 随机种子：`seed=42`  
- 文本预处理：统一做文本清洗（去除 URL/@用户名、合并空白），再交由对应 tokenizer 截断  
- 图像预处理：统一缩放到 224×224，归一化与数值缩放由对应的视觉模型预处理完成  

### BERT+ViT 基线

- 文本编码器：  
  - hidden_dim=512  
  - num_layers=8  
- 融合模块（跨模态注意力）：  
  - hidden_dim=512  
  - cross-attention 结构  
- 分类头：  
  - hidden_dim=256  

### 预训练 ViLBERT（简化版）

- 跨模态 Transformer（fusion encoder）：  
  - hidden_dim=768  
  - fusion_layers=2  
  - fusion_heads=8  
- 分类头：  
  - hidden_dim=512  

### OpenAI CLIP 模型

- 分类头：  
  - cls_hidden_dim 512  

## 实验可复现性说明

- 固定随机性设置：所有训练脚本默认使用 `seed=42`，并在 PyTorch 中设置随机种子。
- 统一训练超参数：关键超参数已在「模型超参数」一节中列出。
- 固定预训练权重版本：BERT / ViT / CLIP 的离线 snapshot 路径在 README 中给出，训练命令示例中通过 `--bert_name` / `--vit_name` / `--clip_model_name` 显式指定，避免因下载到不同版本模型产生偏差。


## 模型结果

### 不同模型架构对比（相近参数量）

| 模型                           | 预训练 | 参数量 (M) | 融合方式    | Val Acc (%) |
|--------------------------------|--------|-----------:|-------------|------------:|
| BERT+ViT（baseline）           | 否     | 125.646    | attn_fuse   | 62          |
| ViLBERT 风格模型               | 是     | 211.033    | attn_fuse   | 67          |
| OpenAI CLIP（clip_fuse）       | 是     | 152.328    | clip_fuse   | 74          |
| OpenAI CLIP（attn_fuse，改进后） | 是   | 163.764    | attn_fuse   | **77**          |

### 同一 OpenAI CLIP 下不同融合策略

OpenAI CLIP 编码器下不同融合策略对比：   

| 编码器 | fusion_type | 模态       | Val Acc (%) |
|--------|-------------|------------|------------:|
| CLIP   | early       | text+image | 76          |
| CLIP   | late        | text+image | 74          |
| CLIP   | attn        | text+image | **77**          |
| CLIP   | clip_match  | text+image | 74          |
| CLIP   | text_only   | text       | 73          |
| CLIP   | image_only  | image      | 62          |


## References

[1] Hamidi M A M, Taqa A Y, Ibrahim Y I. A Systematic Review of Multimodal Sentiment Analysis Based on Text-Image Fusion: Trends, Models, and Research Gaps[J].  
Sinkron: Jurnal dan Penelitian Teknik Informatika, 2025, 9(2).  
DOI: 10.33395/sinkron.v9i2.14840.   
https://jurnal.polgan.ac.id/index.php/sinkron/article/view/14840


[2]
Chang, Y.; Li, Z.; Ruan, Y.; Yin, G. Image–Text Multimodal Sentiment Analysis Algorithm Based on Curriculum Learning and Attention Mechanisms. Big Data Cogn. Comput. 2026, 10, 23. https://doi.org/10.3390/bdcc10010023     

[3]
https://github.com/declare-lab/multimodal-deep-learning


