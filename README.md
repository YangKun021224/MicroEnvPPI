# MicroEnvPPI: Microenvironment-Aware Optimization for Protein–Protein Interaction Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0.0+](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DGL 2.0+](https://img.shields.io/badge/DGL-2.0-orange.svg)](https://www.dgl.ai/)

This repository contains the official PyTorch implementation for our paper:

> **MicroEnvPPI: Microenvironment-Aware Optimization for Protein–Protein Interaction Prediction**
>
> *Kun Yang, Zhen Li, Yifan Chen, Linlin Zhuo*, Yanshi Wei, Haiyang Hu, Dongsheng Cao*, Aiping Lu, Quan Zou and Xiangzheng Fu**
>
> ([Link to Paper - TBD]())

## 摘要

蛋白质-蛋白质相互作用（PPIs）是几乎所有细胞过程的基础。尽管现有计算模型已取得显著进展，但它们往往忽视了残基水平微环境的关键重要性，从而限制了其预测能力和泛化性。

为解决这一问题，我们提出了 **MicroEnvPPI**，这是一个新颖的、以优化残基微环境表示为核心的PPI预测框架。我们的主要贡献和创新点包括：

1.  [cite_start]**更强大的特征表示**: 我们使用强大的蛋白质语言模型 **ESM-2** 的高维上下文感知嵌入，取代了传统的理化特征，极大地丰富了残基表示的语义和进化信息 [cite: 1125, 1396-1401]。
2.  [cite_start]**先进的多任务预训练框架**: 我们设计了一种创新的多任务预训练策略，该策略将基于VQ-VAE的字典学习与两个关键的辅助任务相结合：**图对比学习**和**掩码特征重构** [cite: 1126, 1185, 1486-1490]。这种协同作用显著提升了所学微环境表示的质量和鲁棒性。
3.  [cite_start]**卓越的泛化能力**: 通过联合优化全局PPI推断和局部微环境表示，MicroEnvPPI 表现出卓越的性能和稳定性，尤其是在包含未见蛋白质的挑战性数据集划分（如DFS和BFS）上 [cite: 1128, 1227-1229]。

## 框架概览

MicroEnvPPI框架包括两个主要阶段：(a)一个全面的预训练阶段，用于学习优化的微环境表示；(b)一个下游任务建模阶段，用于最终的PPI预测。

![MicroEnvPPI Framework](https://i.imgur.com/8xYtE9M.png)
*图1: MicroEnvPPI框架概览，详细说明了带有辅助任务的预训练和下游PPI建模。*

## 安装

#### 1. 克隆代码库

```bash
git clone [https://github.com/your-username/MicroEnvPPI.git](https://github.com/your-username/MicroEnvPPI.git)
cd MicroEnvPPI
```

#### 2. 创建并激活Conda环境

我们强烈建议使用Conda来管理项目依赖。您可以从我们提供的 `environment.yml` 文件创建环境。

```bash
# 推荐使用conda进行环境管理
conda env create -f environment.yml
conda activate MicroEnvPPI
```

*（注意：如果 `environment.yml` 文件缺失，您可以根据 `import` 语句手动创建一个，关键依赖包括 `python=3.8`, `pytorch`, `dgl`, `transformers`, `scikit-learn` 等。）*

## 数据准备

数据处理流程涉及三个主要步骤，请按顺序执行：

#### 步骤 1: 下载原始数据

-   从[STRING数据库](https://string-db.org/)下载原始的蛋白质序列（`protein.sequences.dictionary.tsv`）、PPI关系（`protein.actions.txt`）文件。
-   从[AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)下载对应物种的蛋白质结构PDB文件。
-   将下载的序列和关系文件放入项目根目录下的 `raw_data/` 文件夹。
-   在 `raw_data/` 内创建一个名为 `STRING_AF2DB/` 的子文件夹，并将所有PDB文件放入其中。

#### 步骤 2: 生成ESM-2嵌入

-   [cite_start]我们的模型使用ESM-2（650M版本）的1280维残基嵌入作为初始节点特征 [cite: 70, 77, 1397, 1398]。
-   首先，请从Hugging Face Hub下载`esm2_t33_650M_UR50D`模型权重，或确保您的环境可以访问它。
-   运行`generate_esm_embeddings.py`脚本来为您的数据集生成特征文件。**请务必在使用前修改脚本内的`dataset`和`local_model_path`变量**。

```bash
# 进入src目录
cd src

# 为SHS148k数据集生成嵌入（示例）
python generate_esm_embeddings.py
```
-   该脚本会生成一个如 `data/processed_data/protein.nodes.esm2_650M.SHS148k.pt` 的文件。

#### 步骤 3: 处理图结构数据

-   在生成了ESM嵌入之后，运行`data_process.py`脚本来处理PDB文件，提取基于三维空间的边（半径邻居和k近邻）。

```bash
# 仍在src目录下
python data_process.py --dataset SHS148k
```
-   此脚本会生成`protein.rball.edges.SHS148k.npy`和`protein.knn.edges.SHS148k.npy`等文件，存放于`data/processed_data/`目录。

## 如何运行模型

`train.py`脚本是执行预训练和下游任务建模的统一入口。

### 1. 完整流程：预训练 + 下游任务

要从头开始运行包括预训练在内的完整流程，请使用以下命令。脚本将自动从`configs/param_configs.json`加载为不同数据集和划分方式优化的超参数。

```bash
# 仍在src目录下
python train.py --dataset SHS148k --split_mode bfs --seed 42
```
-   `--dataset`: 指定数据集 (`SHS27k`, `SHS148k`, `STRING`)。
-   [cite_start]`--split_mode`: 指定数据划分方式 (`random`, `bfs`, `dfs`) [cite: 252]。
-   `--seed`: 设置随机种子以保证结果可复现。

预训练好的VAE模型 (`vae_model.ckpt`) 将保存在`results/`目录下的对应时间戳文件夹中。

### 2. 仅下游任务：使用已有的预训练模型进行评估

如果您已经有了一个预训练好的`vae_model.ckpt`，您可以通过`--ckpt_path`参数指定其路径，从而**跳过耗时的预训练阶段**，直接进行下游GIN模型的训练和评估。

```bash
# 仍在src目录下
python train.py \
    --dataset SHS148k \
    --split_mode bfs \
    --ckpt_path ../results/SHS148k/your_timestamp_folder/VAE_CL_Aux_RandMCM/vae_model.ckpt
```

### 3. 断点续训

本脚本支持从检查点（checkpoint）恢复训练，无论是VAE预训练还是GIN下游任务训练。请使用`--resume`标志，并指向对应的检查点文件。

```bash
# 示例：恢复VAE预训练
python train.py --dataset STRING --split_mode random --resume ../results/STRING/.../vae_cl_aux_randmcm_checkpoint.pth

# 示例：恢复GIN下游任务训练
python train.py --dataset STRING --split_mode random --resume ../results/STRING/.../gin_cl_aux_randmcm_checkpoint.pth
```

## 引用

如果我们的工作对您的研究有所帮助，请考虑引用我们的论文：

```bibtex
@article{yang2024microenvppi,
  title={MicroEnvPPI: Microenvironment-Aware Optimization for Protein–Protein Interaction Prediction},
  author={Yang, Kun and Li, Zhen and Chen, Yifan and Zhuo, Linlin and Wei, Yanshi and Hu, Haiyang and Cao, Dongsheng and Lu, Aiping and Zou, Quan and Fu, Xiangzheng},
  journal={TBD},
  year={2024}
}
```

## 联系方式

如果您有任何问题或建议，欢迎通过GitHub Issue与我们交流，或直接联系通讯作者：
- **Linlin Zhuo**: 20210339@wzut.edu.cn
- **Dongsheng Cao**: oriental-cds@163.com
- **Xiangzheng Fu**: fxzheng@hkbu.edu.hk
```
