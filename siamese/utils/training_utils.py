import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from siamese.configs.config import Config

def get_callbacks(model_name, config=None):
    """
    Creates callbacks for model training.
    
    Parameters:
        model_name: Name of the model
        config: Configuration object
        
    Returns:
        List of callbacks
    """
    if config is None:
        config = Config()
    
    # Create directories if they don't exist
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Model checkpoint to save the best model
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    
    # Reduce learning rate when the model plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        mode='min',
        min_lr=1e-6
    )
    
    return [checkpoint, early_stopping, reduce_lr]

def plot_training_history(history, model_name, config=None):
    """
    Plots the training and validation loss and accuracy.
    
    Parameters:
        history: Training history object
        model_name: Name of the model
        config: Configuration object
    """
    if config is None:
        config = Config()
    
    # Create the results directory if it doesn't exist
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    if isinstance(history, dict):
        history_dict = history
    else:
        history_dict = history.history
    
    # Plot based on available metrics
    metrics_to_plot = []
    
    # Check if we have standard metrics
    if 'loss' in history_dict and 'val_loss' in history_dict:
        metrics_to_plot.append(('loss', 'val_loss', 'Loss'))
        
    if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
        metrics_to_plot.append(('accuracy', 'val_accuracy', 'Accuracy'))
    
    # Check if we have joint training metrics
    if 'siamese_loss' in history_dict and 'class_loss' in history_dict:
        metrics_to_plot.extend([
            ('siamese_loss', 'siamese_val_loss', 'Siamese Loss'),
            ('class_loss', 'class_val_loss', 'Classification Loss'),
            ('joint_loss', 'joint_val_loss', 'Joint Loss')
        ])
        
        if 'siamese_acc' in history_dict and 'class_acc' in history_dict:
            metrics_to_plot.extend([
                ('siamese_acc', 'siamese_val_acc', 'Siamese Accuracy'),
                ('class_acc', 'class_val_acc', 'Classification Accuracy')
            ])
    
    # Create plot grid
    num_metrics = len(metrics_to_plot)
    if num_metrics == 0:
        print("No metrics to plot")
        return
    
    # Create figure with proper size
    num_rows = (num_metrics + 1) // 2  # Ceil division
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    
    # Make axes a 2D array if it's 1D
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    # Plot each metric
    for idx, (train_metric, val_metric, title) in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        
        ax = axes[row, col]
        
        if train_metric in history_dict:
            ax.plot(history_dict[train_metric], label=f'Training {title}')
        if val_metric in history_dict:
            ax.plot(history_dict[val_metric], label=f'Validation {title}')
            
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
    
    # Hide any unused subplots
    for idx in range(num_metrics, num_rows * 2):
        row = idx // 2
        col = idx % 2
        fig.delaxes(axes[row, col])
    
    plt.suptitle(f'{model_name} Training History')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, f'{model_name}_training_history.png'))
    plt.close()

def evaluate_classification_model(model, val_data, class_names=None, config=None):
    """
    Evaluates the classification model and returns metrics.
    
    Parameters:
        model: Classification model to evaluate
        val_data: Validation data generator or dataset
        class_names: List of class names
        config: Configuration object
        
    Returns:
        Dictionary of evaluation metrics
    """
    if config is None:
        config = Config()
        
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    # Get predictions from the model
    if hasattr(val_data, 'reset'):
        val_data.reset()
        
    # Predict on validation data
    y_true = []
    y_pred = []
    
    if hasattr(val_data, 'as_numpy_iterator'):
        # It's a TensorFlow dataset
        for batch_X, batch_y in val_data:
            batch_pred = model.predict(batch_X)
            
            # Convert from one-hot to class indices
            if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                batch_y = np.argmax(batch_y, axis=1)
                
            batch_pred = np.argmax(batch_pred, axis=1)
            
            y_true.extend(batch_y)
            y_pred.extend(batch_pred)
    else:
        # It's a Keras data generator
        if hasattr(val_data, 'classes'):
            # If it's a directory iterator, we can get classes directly
            y_true = val_data.classes
            
            # Make predictions on the entire dataset
            predictions = model.predict(val_data)
            y_pred = np.argmax(predictions, axis=1)
        else:
            # We need to iterate through the data
            num_samples = len(val_data) * val_data.batch_size
            steps = len(val_data)
            
            for i in range(steps):
                batch_X, batch_y = val_data[i]
                batch_pred = model.predict(batch_X)
                
                # Convert from one-hot to class indices
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    batch_y = np.argmax(batch_y, axis=1)
                    
                batch_pred = np.argmax(batch_pred, axis=1)
                
                y_true.extend(batch_y)
                y_pred.extend(batch_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config.CONFUSION_MATRIX_PATH), exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    plt.close()
    
    # Return metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return metrics

def enable_mixed_precision(config=None):
    """
    Enables mixed precision training for faster training with less memory.
    
    Parameters:
        config: Configuration object
    """
    if config is None:
        config = Config()
        
    if config.ENABLE_MIXED_PRECISION:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print('Mixed precision enabled')
        except:
            print("Mixed precision not supported on this hardware, continuing with default precision")

def set_random_seed(seed=None, config=None):
    """
    Sets random seeds for reproducibility.
    
    Parameters:
        seed: Random seed
        config: Configuration object
    """
    if config is None:
        config = Config()
        
    if seed is None:
        seed = config.RANDOM_SEED
        
    np.random.seed(seed)
    tf.random.set_seed(seed) 