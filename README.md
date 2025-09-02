# LargeDeepSurvival: 3D ResNet Pretraining + Survival Fine-tuning for DICOM Images

This repository provides a robust, scalable pipeline for pretraining 3D ResNet models on large-scale DICOM image datasets and fine-tuning them for survival analysis (time-to-event) with Cox proportional hazards. It is designed for speed, flexibility, and best practices in deep learning research.

## Features
- **3D ResNet architectures** (ResNet18/34/50/101/152) via MONAI
- **Flexible dataset loading** with support for both classic PyTorch and MONAI-style caching
- **DICOM image preprocessing** with resizing, normalization, and augmentations
- **Multi-GPU support** via PyTorch `DataParallel`
- **TensorBoard integration** for real-time monitoring
- **Efficient for large datasets** (100,000+ images)
- **Comprehensive test suite** for all modules

---

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd LargeDeepSurvival
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Example with conda (recommended)
   conda create -n torchsurv2 python=3.10 -y
   conda activate torchsurv2
   # Or use venv if preferred
   # python3 -m venv venv && source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional but recommended) **Install TorchSurv for metrics**
   ```bash
   pip install torchsurv
   ```
   When installed, evaluation uses TorchSurv’s optimized metrics implementations (C-index, Integrated Brier Score, time-dependent AUC). Otherwise, built-in fallbacks are used.

---

## Usage

### 1. Pretraining
Pretrain a 3D ResNet to predict tabular features (e.g., clinical variables, Agatston scores, calcium-omics) directly from DICOM images:
```bash
python pretraining.py \
  --csv_path your_data.csv \
  --feature_cols feature1 feature2 \
  --resnet resnet18 \
  --batch_size 8 \
  --epochs 100 \
  --image_size 256 256 64
```
- **TensorBoard logs** are saved in `pretrain_logs/`. Launch TensorBoard with:
  ```bash
  tensorboard --logdir pretrain_logs
  ```
- **Model checkpoints** and best model are saved in `pretrain_logs/`.

### 2. Fine-tuning for Survival (CoxPH)
Train a Cox proportional hazards head for time-to-event prediction and compare initialization modes (random vs pretraining):
```bash
# Pretrain to predict tabular features (optional but recommended)
python pretraining.py \
  --csv_path train_pretrain.csv \
  --feature_cols clinical1 clinical2 agatston calcium_omics1 \
  --resnet resnet18 \
  --batch_size 8 \
  --epochs 50 \
  --image_size 256 256 64 \
  --output pretrain_logs/pretrained_resnet.pth

# Fine-tune with random init
python finetuning.py \
  --train_csv train.csv \
  --val_csv val.csv \
  --init random \
  --resnet resnet18 \
  --batch_size 6 \
  --epochs 50 \
  --image_size 256 256 64 \
  --eval_years 1 2 3 4 5 \
  --log_dir finetune_logs/random

# Fine-tune with pretrained init
python finetuning.py \
  --train_csv train.csv \
  --val_csv val.csv \
  --init pretrained \
  --pretrained_path pretrain_logs/pretrained_resnet.pth \
  --resnet resnet18 \
  --batch_size 6 \
  --epochs 50 \
  --image_size 256 256 64 \
  --eval_years 1 2 3 4 5 \
  --log_dir finetune_logs/pretrained
```
- Compare validation **C-index**, **Integrated Brier Score (IBS)**, and **time-dependent AUC** at 1–5 years to test the hypothesis that pretraining improves downstream survival performance.

### 3. Testing
Run the test suite to verify all modules:
```bash
python test_network.py           # Test ResNet initialization and forward pass
python test_dataset.py           # Test dataset loading and output
python test_preprocessing.py     # Test CSV and feature preprocessing
python test_monai_survival_dataset.py  # End-to-end dataset test with DICOMs
```

---

## Data Format
- **CSV file** should contain columns:
  - `Dicom path`: Path to a DICOM series (directory of DICOM slices)
  - `time`, `event`: Survival analysis labels
  - `feature1`, `feature2`, ...: Features to predict (can be clinical, radiomic, etc.)

Notes:
- Ensure your `time` units are consistent. If you evaluate at 1–5 years, provide `time` in years (or adjust `--eval_years` to your units, e.g., days).
- `event` is 1 if the event occurred, 0 if censored.

---

## Metrics (TorchSurv-aligned)
When `torchsurv` is installed, `finetuning.py` computes metrics using TorchSurv’s implementations; otherwise, built-in fallbacks are used.

- **C-index (Concordance Index)**: Measures pairwise concordance between predicted risk and observed times. Concordant if the patient with the earlier event has higher predicted risk; ties count as 0.5; non-comparable pairs are ignored. Implemented via `torchsurv.metrics.ConcordanceIndex`.

- **Integrated Brier Score (IBS)**: Measures calibrated prediction error of the predicted survival functions over time, using IPCW (inverse probability of censoring weighting) with KM censoring estimator. Implemented via `torchsurv.metrics.BrierScore` where we call it on survival probabilities across a grid of evaluation times and report `.integral()`.

- **Time-dependent AUC (tAUC)**: Discrimination at specific times using IPCW. Implemented via `torchsurv.metrics.Auc`; we report values at `--eval_years` (default 1–5 years).

Implementation details:
- The survival head outputs log-risk scores. We estimate the Cox baseline cumulative hazard via Breslow on validation predictions and compute survival probabilities `S(t) = exp(-H0(t) * exp(lp))` at requested times.
- Metrics are printed each epoch for validation: `C-index`, `IBS`, and `tAUC@[years]`.

If `torchsurv` is unavailable, internal, numerically stable PyTorch implementations are used with the same semantics.

## Best Practices & Tips
- For large datasets, set `use_cache=False` in `MonaiSurvivalDataset` to avoid running out of RAM.
- Use as many `num_workers` as your CPU allows for best DataLoader speed.
- Monitor training with TensorBoard for real-time feedback.
- Checkpoints are saved automatically on best validation C-index.
- All code is modular and easily extensible for pretraining tasks, fine-tuning heads, or additional metrics.

## Experiment: Testing the Pretraining Hypothesis
- Train two models on the same train/val split:
  - **Random init**: `--init random`
  - **Pretrained init**: `--init pretrained --pretrained_path pretrain_logs/pretrained_resnet.pth`
- Compare validation metrics: higher C-index, higher tAUC, and lower IBS indicate better performance. If pretraining helps, the pretrained run should outperform random init.

---

## Citation & Acknowledgements
- Built on [MONAI](https://monai.io/) and [PyTorch](https://pytorch.org/).
- DICOM handling via [pydicom](https://pydicom.github.io/).
 - Survival metrics via [TorchSurv](https://opensource.nibr.com/torchsurv/) when installed.

---

## License
MIT License 