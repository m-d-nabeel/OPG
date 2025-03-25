# Siamese OPG Age Classification

This project implements a sophisticated approach to age classification from Orthopantomogram (OPG) images using a combination of Siamese Networks and Classification models with shared feature extractors.

## Features

- Joint training of Siamese and Classification networks with shared feature extractors
- Multiple feature extractor options: CNN, Vision Transformer (ViT), or Hybrid
- Image similarity comparison using Siamese networks
- Age range classification with high accuracy
- Find similar images in a dataset based on feature similarity
- Modular and extensible architecture

## Project Structure

```
siamese/
├── configs/              # Configuration settings
├── data/                 # Data handling utilities
├── models/               # Model architectures
│   ├── feature_extractors.py  # Feature extraction models (CNN, ViT, Hybrid)
│   ├── siamese_model.py       # Siamese network implementation
│   ├── classification_model.py # Classification model implementation
│   └── joint_model.py         # Joint training model
├── utils/                # Utility functions
├── train/                # Training scripts
├── inference/            # Inference and prediction tools
├── checkpoints/          # Saved models (created during training)
└── results/              # Training results and visualizations
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/siamese-opg.git
cd siamese-opg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

You can train the models in two ways:

1. **Joint Training** (recommended): Train the Siamese network and classification model together with shared weights:

```bash
python -m siamese.train.train --mode joint --extractor hybrid
```

2. **Separate Training**: Train the Siamese network first, then use its feature extractor for the classification model:

```bash
python -m siamese.train.train --mode separate --extractor hybrid
```

Training options:

- `--mode`: Training mode (`joint` or `separate`)
- `--extractor`: Type of feature extractor (`cnn`, `vit`, or `hybrid`)
- `--train_dir`: Path to training data directory
- `--val_dir`: Path to validation data directory
- `--batch_size`: Batch size for training
- `--epochs`: Number of epochs for training
- `--learning_rate`: Learning rate for training

### Inference

For predicting age ranges from OPG images:

```bash
# Predict age range for a single image
python -m siamese.inference.predict predict --image path/to/image.jpg

# Batch prediction for multiple images
python -m siamese.inference.predict batch --dir path/to/images/

# Find similar images
python -m siamese.inference.predict similar --query path/to/query.jpg --dir path/to/dataset/

# Compare two images
python -m siamese.inference.predict compare --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

## Dataset Structure

The expected dataset structure is:

```
data/
├── train/
│   ├── Age_5-10/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Age_11-14/
│   ├── Age_15-18/
│   └── Age_19-24/
└── val/
    ├── Age_5-10/
    ├── Age_11-14/
    ├── Age_15-18/
    └── Age_19-24/
```

## Key Components

### Feature Extractors

Three types of feature extractors are available:

1. **CNN**: Uses MobileNetV2 pre-trained on ImageNet
2. **ViT**: Vision Transformer implementation
3. **Hybrid**: Combines CNN and ViT features for better performance

### Siamese Network

The Siamese network uses shared feature extractors to learn similarity between images. It's trained to distinguish whether two images belong to the same age class or different classes.

### Classification Model

The classification model takes the learned feature extractor from the Siamese network to classify images into age ranges. It includes an attention mechanism to focus on the most relevant features.

### Joint Training

The joint training approach trains both models simultaneously, allowing them to benefit from each other's learning signals. This is implemented using a custom training loop with a weighted loss function.

## Customization

You can customize the models by modifying the configuration in `siamese/configs/config.py`. Key parameters include:

- Image dimensions
- Model hyperparameters
- Data augmentation settings
- Training settings

## Requirements

- Python 3.7+
- TensorFlow 2.4+
- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Vision Transformer implementation is inspired by the original ViT paper
- MobileNetV2 architecture from TensorFlow/Keras
- The Siamese Network approach is based on the original concept by Bromley et al. 