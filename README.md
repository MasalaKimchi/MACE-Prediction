# Survival Analysis Project

A compartmentalized survival analysis framework for medical imaging, inspired by the [SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D) architecture structure. This project provides a clean, modular approach to survival analysis using 3D medical imaging data with state-of-the-art deep learning architectures.

## Project Structure

```
SurvivalProject/
├── architectures/          # Model architectures and neural network implementations
│   ├── __init__.py
│   ├── README.md          # Architecture documentation
│   └── resnet3d.py        # 3D ResNet implementations (ResNet18, 34, 50, 101, 152)
├── dataloaders/           # Data loading and dataset classes
│   ├── __init__.py
│   ├── README.md          # Data loading documentation
│   └── survival_dataset.py # Survival analysis datasets with MONAI integration
├── data/                  # Data preprocessing utilities
│   ├── __init__.py
│   ├── README.md          # Data preprocessing documentation
│   └── preprocessing.py   # Data preprocessing functions and pipelines
├── losses/                # Loss functions for survival analysis
│   ├── __init__.py
│   ├── README.md          # Loss functions documentation
│   └── cox_loss.py        # Cox proportional hazards loss implementation
├── metrics/               # Evaluation metrics for survival analysis
│   ├── __init__.py
│   ├── README.md          # Metrics documentation
│   └── survival_metrics.py # C-index, Brier score, time-dependent AUC, etc.
├── optimizers/            # Optimizer configurations and factory
│   ├── __init__.py
│   ├── README.md          # Optimizer documentation
│   └── optimizer_factory.py # Optimizer factory with multiple options
├── augmentations/         # Data augmentation transforms
│   ├── __init__.py
│   ├── README.md          # Augmentation documentation
│   └── transforms.py      # MONAI-based image augmentation pipelines
├── experiments/           # Experiment configurations and templates
│   └── survival_analysis/
│       └── template_experiment/
│           ├── config.yaml      # YAML-based experiment configuration
│           ├── run_experiment.py # Config-driven training script
│           └── README.md        # Experiment documentation
├── train_scripts/         # Standalone training scripts
│   ├── finetune_survival.py    # Fine-tuning script for survival models
│   └── pretrain_features.py    # Pre-training script for feature extraction
├── tests/                 # Comprehensive test suite
│   ├── __init__.py
│   ├── README.md          # Testing documentation
│   ├── run_all_tests.py   # Test runner
│   ├── test_architectures.py # Architecture tests
│   ├── test_data_preprocessing.py # Preprocessing tests
│   └── test_dataloaders.py # Data loading tests
├── __init__.py           # Main package imports and exports
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
└── README.md            # This file
```

## Features

- **Compartmentalized Architecture**: Clean separation of concerns following [SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D) structure
- **3D ResNet Models**: Support for ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 with 3D convolutions
- **Survival Analysis**: Cox proportional hazards model with comprehensive metrics using [torchsurv](https://github.com/autonlab/torchsurv)
- **Medical Imaging**: [MONAI](https://monai.io/) based 3D medical image processing and augmentation
- **Config-based Training**: YAML configuration for reproducible experiments
- **Comprehensive Metrics**: C-index, Brier score, time-dependent AUC, integrated Brier score
- **Modular Design**: Easy to extend with new architectures, datasets, and loss functions
- **Testing Suite**: Comprehensive unit tests for all components

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SurvivalProject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Activate the conda environment (if using):
```bash
conda activate torchsurv2
```

## Usage

### Config-based Training

1. Copy the template experiment:
```bash
cp -r experiments/survival_analysis/template_experiment experiments/survival_analysis/my_experiment
```

2. Edit the configuration file:
```bash
nano experiments/survival_analysis/my_experiment/config.yaml
```

3. Run the experiment:
```bash
cd experiments/survival_analysis/my_experiment
python run_experiment.py --config config.yaml
```

### Standalone Scripts

#### Pre-training
```bash
python train_scripts/pretrain_features.py \
    --csv_path data/train.csv \
    --feature_cols feature1 feature2 feature3 \
    --resnet resnet18 \
    --batch_size 8 \
    --epochs 100
```

#### Fine-tuning
```bash
python train_scripts/finetune_survival.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --resnet resnet18 \
    --batch_size 6 \
    --epochs 50 \
    --lr 1e-4
```

## Configuration

The `config.yaml` file allows you to configure:

- **Dataset**: CSV paths, column names, image size, augmentation
- **Model**: Architecture, initialization, pretrained weights
- **Training**: Batch size, epochs, learning rate, optimizer
- **Evaluation**: Time points for metrics calculation
- **Logging**: Output paths, tensorboard, wandb integration

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **[MONAI](https://monai.io/)**: Medical imaging AI toolkit for 3D image processing
- **[torchsurv](https://github.com/autonlab/torchsurv)**: Survival analysis library for PyTorch
- **Polars**: Fast DataFrame library for data processing
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities

### Additional Dependencies
- **PyYAML**: YAML configuration parsing
- **TensorBoard**: Experiment logging and visualization
- **PyDICOM**: DICOM medical image format support
- **Matplotlib**: Plotting and visualization
- **Pillow**: Image processing

## Detailed Folder Documentation

Each major folder contains its own README.md with detailed documentation:

- **[architectures/README.md](architectures/README.md)**: Model architectures and neural network implementations
- **[dataloaders/README.md](dataloaders/README.md)**: Data loading and dataset classes
- **[data/README.md](data/README.md)**: Data preprocessing utilities and pipelines
- **[losses/README.md](losses/README.md)**: Loss functions for survival analysis
- **[metrics/README.md](metrics/README.md)**: Evaluation metrics and scoring functions
- **[optimizers/README.md](optimizers/README.md)**: Optimizer configurations and factory
- **[augmentations/README.md](augmentations/README.md)**: Data augmentation transforms
- **[experiments/README.md](experiments/README.md)**: Experiment configurations and templates
- **[tests/README.md](tests/README.md)**: Testing documentation and guidelines

## Contributing

1. Follow the compartmentalized structure inspired by [SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D)
2. Add new architectures to `architectures/` with corresponding tests
3. Add new datasets to `dataloaders/` with MONAI integration
4. Add new losses to `losses/` compatible with torchsurv
5. Add new metrics to `metrics/` for survival analysis
6. Update `__init__.py` files to expose new functionality
7. Add comprehensive tests for new components
8. Update relevant README.md files with documentation

## Acknowledgments

This project is inspired by and follows the architectural patterns from:
- **[SegFormer3D](https://github.com/OSUPCVLab/SegFormer3D)**: Official Implementation of SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation (CVPR/W 2024)
- **[torchsurv](https://github.com/autonlab/torchsurv)**: Survival analysis library for PyTorch
- **[MONAI](https://monai.io/)**: Medical imaging AI toolkit for 3D image processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

