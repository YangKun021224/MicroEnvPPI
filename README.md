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

## Abstract

Protein-protein interactions (PPIs) are fundamental to almost all cellular processes. To address the limitations of existing computational models in characterizing residue microenvironments, we propose MicroEnvPPI, a novel PPI prediction framework that focuses on optimizing microenvironment representation. This work significantly improves prediction accuracy and generalization capability on challenging datasets by integrating powerful ESM-2 language model embeddings and innovative multi-task self-supervised pretraining strategies (including graph contrastive learning).

## 框架概览

MicroEnvPPI 通过一个两阶段框架优化残基微环境表示，以实现高精度的PPI预测。

![MicroEnvPPI Framework](MicroEnvPPI/assets/framework.png)

*Figure 1: Overview of the MicroEnvPPI framework, detailing pretraining with auxiliary tasks and downstream PPI modeling.*

## Framework Overview

```
MicroEnvPPI/
├── assets/        
├── configs/             
│   └── param_configs.json
├── data/
│   └── processed_data/     
├── raw_data/               
│   └── STRING_AF2DB/
├── results/                
├── src/                   
│   ├── data_process.py   
│   ├── dataloader.py       
│   ├── generate_esm_embeddings.py 
│   ├── models.py          
│   ├── train.py           
│   └── utils.py            
├── environment.yml        
└── README.md               
```

## Quick Start: Evaluation with Pre-trained Models

We provide pre-trained models on the SHS27k dataset for three different splitting methods: random, bfs, and dfs. You can follow the steps below to quickly reproduce evaluation results.

#### 1. Setup Environment and Code

```bash
# Clone this repository
git clone [https://github.com/yangkun021224/MicroEnvPPI.git](https://github.com/yangkun021224/MicroEnvPPI.git)
cd MicroEnvPPI

# Create and activate environment using Conda
conda env create -f environment.yml
conda activate MicroEnvPPI
```

#### 2. Download Data and Pre-trained Models

-   **Required: Download processed data**
    -   We strongly recommend downloading our processed data directly to skip the tedious data preprocessing steps.
    -   Download link: [processed_data.zip (Google Drive)](https://drive.google.com/file/d/1mWrgzMxuHHIMsDA2OL8r0lNShiCUWc6Y/view?usp=drive_link)
    -   After downloading, please extract and place the obtained processed_data folder in the data/ folder under the project root directory.

-   **Required: Download pre-trained models**
    -   我们所有的实验结果和模型检查点都已上传。
    -   Download link:  [results (Google Drive)](https://drive.google.com/file/d/1lR8WeZTQMwOSnUFiruShmYzyPBiNJFmg/view?usp=drive_link)
    -   After downloading, please extract and place the obtained results folder in the project root directory.

#### 3. Run Evaluation Commands

After downloading and placing the above files, you can directly run the following commands to evaluate the corresponding pre-trained models:

```bash

cd src

# Evaluate on SHS27k (random split)
python train.py --dataset SHS27k --split_mode random --ckpt_path "../results/SHS27k/2025-04-29_17-21-12_279/VAE_CL_Aux_RandMCM/vae_model.ckpt"

# Evaluate on SHS27k (bfs split)
python train.py --dataset SHS27k --split_mode bfs --ckpt_path "../results/SHS27k/2025-04-30_01-13-55_572/VAE_CL_Aux_RandMCM/vae_model.ckpt"

# Evaluate on SHS27k (dfs split)
python train.py --dataset SHS27k --split_mode dfs --ckpt_path "../results/SHS27k/2025-04-29_18-34-09_183/VAE_CL_Aux_RandMCM/vae_model.ckpt"
```


---

## Training from Scratch

If you wish to start from raw data and fully reproduce our data processing and model training pipeline, please follow these steps.

### 1. Environment Setup
（同上文“快速开始”部分）

### 2. Data Preparation

-   **Download raw data**:
    -   Download link: [raw_data.rar (Google Drive)](https://drive.google.com/file/d/1nq5UZIhkrMUsS_N4oVKs5l3fM82JsFZl/view?usp=drive_link)
    -   After downloading, extract and place all contents in the raw_data/ folder under the project root directory. Ensure it contains the STRING_AF2DB subfolder with PDB files.

-   **Generate ESM-2 embeddings**:
    -  Run src/generate_esm_embeddings.py to generate initial features for your dataset.
    -  **Note**: Please make sure to modify the dataset and local_model_path variables in the script before running.
    ```bash
    cd src
    python generate_esm_embeddings.py
    ```

-   **Process graph structure data**:
    -   Run src/data_process.py to process PDB files and generate graph edge files.
    ```bash
    # Still in the src directory
    python data_process.py --dataset <dataset_name：SHS27k,SHS148k,STRING>
    ```

### 3. Run Training

-   **Pretraining + Downstream task**:
    -   To run the complete training pipeline (VAE pretraining followed by downstream GIN model training), execute:
    ```bash
    # Still in the src directory
    python train.py --dataset SHS148k --split_mode bfs --seed 42
    ```
    -  After training completion, the optimal VAE model vae_model.ckpt and GIN model model_..._best_state.pth will be saved in the results/ directory.

-   **Resume training**:
    -   If training is interrupted unexpectedly, you can use the --resume parameter to resume from checkpoints.
    ```bash
    # Example: Resume GIN downstream task training
    python train.py --dataset STRING --split_mode random --resume ../results/STRING/.../gin_cl_aux_randmcm_checkpoint.pth
    ```

## Citation

If our work is helpful to your research, please consider citing our paper:

```bibtex
@article{yang2024microenvppi,
  title={MicroEnvPPI: Microenvironment-Aware Optimization for Protein–Protein Interaction Prediction},
  author={Yang, Kun and Li, Zhen and Chen, Yifan and Zhuo, Linlin and Wei, Yanshi and Hu, Haiyang and Cao, Dongsheng and Lu, Aiping and Zou, Quan and Fu, Xiangzheng},
  journal={TBD},
  year={2024}
}
```

## Contact

If you have any questions or suggestions, please feel free to communicate with us through GitHub Issues, or contact the corresponding authors directly:
- **Linlin Zhuo**: 20210339@wzut.edu.cn
- **Dongsheng Cao**: oriental-cds@163.com
- **Xiangzheng Fu**: fxzheng@hkbu.edu.cn
```
