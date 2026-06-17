# [MIDL 2025] Chest-OMDL: Organ-specific Multidisease Detection and Localization in Chest Computed Tomography using Weakly Supervised Deep Learning from Free-text Radiology Report

**Authors:** Xuguang Bai*, Mingxuan Liu*, Yifei Chen, Hongjia Yang, Qiyuan Tian

**Conference:** Medical Imaging with Deep Learning (MIDL) 2025

[[Paper Link](https://openreview.net/pdf?id=ns6nq592HX)]

---

## 🛠️ Environment Setup

We recommend using Anaconda to manage the environment. Please follow the steps below strictly to avoid installation issues with `mamba-ssm`.

### 1. Create and Activate Environment
```bash  
conda create -n omdl python=3.10  
conda activate omdl  
```

### 2. Install PyTorch (CUDA 12.1)
Ensure your CUDA version is compatible.
```bash  
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121  
```

### 3. Install Basic Dependencies
```bash  
pip install -r requirements.txt  
```

### 4. Install Mamba-SSM (Crucial Step)
`mamba-ssm` requires compilation. We use `--no-build-isolation` to ensure it uses the installed PyTorch and CUDA environment.

```bash  

pip install --no-build-isolation mamba-ssm==2.2.4  
```

> **Note:** If you encounter any `ModuleNotFoundError` during runtime, please install the missing packages manually using pip (e.g., `pip install package_name`).

---

## 📂 Data Preprocessing

### 1. Training & In-Distribution Testing Data
For the standard training and testing dataset, use the following script:

```bash  
python Chest_OMDL/data_preprocessed/data_preprocessed.py  
```

### 2. Out-of-Distribution (OOD) / External Data
For external datasets or data with different distributions, the preprocessing consists of two steps:

**Step 1: Basic Preprocessing**
```bash  
python Chest_OMDL/data_preprocessed/new_data_preprocessed.py  
```

**Step 2: Orientation Adjustment**
To align the data orientation with the training set, run the flip script.
*Note: This is applicable if the original data orientation matches `Chest_OMDL/example_data/val_data.nii.gz`.*

```bash  
python Chest_OMDL/data_preprocessed/flip_data.py  
```
After these two steps, the data is ready for inference.

---

## 🚀 Usage

### Training
To train the model:

```bash  
python train.py  
```

### Testing / Inference
To run inference on the processed data:

```bash  
python test.py  
```

---

## 📝 Citation

If you find this project useful, please cite our paper:

```bibtex  
@inproceedings{bai2025chestomdl,
title={Chest-{OMDL}: Organ-specific Multidisease Detection and Localization in Chest Computed Tomography using Weakly Supervised Deep Learning from Free-text Radiology Report},
author={Xuguang Bai and Mingxuan Liu and Yifei Chen and Hongjia Yang and Qiyuan Tian},
booktitle={Medical Imaging with Deep Learning},
year={2025},
url={https://openreview.net/forum?id=ns6nq592HX}
}
