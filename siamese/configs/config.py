import os

class Config:
    # Paths and basic settings
    TRAIN_DIR = os.path.join('data', 'train')
    VAL_DIR = os.path.join('data', 'val')
    IMG_HEIGHT, IMG_WIDTH = 1024, 512
    BATCH_SIZE = 8  # Reduced batch size due to ViT memory requirements
    EPOCHS = 50
    LEARNING_RATE = 0.0001

    # Model Settings
    PATCH_SIZE = 32  # Must divide image dimensions evenly
    NUM_PATCHES = (IMG_HEIGHT // PATCH_SIZE) * (IMG_WIDTH // PATCH_SIZE)
    PROJECTION_DIM = 128  # Embedding dimension
    NUM_HEADS = 8
    TRANSFORMER_LAYERS = 4
    MLP_HEAD_UNITS = [512, 256]
    FEATURE_DIMENSION = 256  # Output dimension of feature extractor

    # Class Names
    CLASS_NAMES = ['Age_5-10', 'Age_11-14', 'Age_15-18', 'Age_19-24']
    NUM_CLASSES = len(CLASS_NAMES)

    # Augmentation Parameters
    ROTATION_RANGE = 5
    WIDTH_SHIFT_RANGE = 0.05
    HEIGHT_SHIFT_RANGE = 0.05
    BRIGHTNESS_RANGE = [0.9, 1.1]
    ZOOM_RANGE = 0.05
    HORIZONTAL_FLIP = False
    FILL_MODE = 'constant'
    CVAL = 0

    # Training Settings
    ENABLE_MIXED_PRECISION = True
    RANDOM_SEED = 42
    
    # Model Checkpoints
    CHECKPOINT_DIR = 'checkpoints'
    SIAMESE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'siamese_model_best.keras')
    CLASSIFICATION_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'classification_model_best.keras')
    FEATURE_EXTRACTOR_PATH = os.path.join(CHECKPOINT_DIR, 'feature_extractor_best.keras')
    
    # Results
    RESULTS_DIR = 'results'
    CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.png')
    DATASET_DISTRIBUTION_PATH = os.path.join(RESULTS_DIR, 'dataset_distribution.png') 