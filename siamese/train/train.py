import os
import argparse
import tensorflow as tf
from siamese.configs.config import Config
from siamese.data.data_utils import DataManager
from siamese.models.siamese_model import SiameseNetwork
from siamese.models.classification_model import ClassificationModel
from siamese.models.joint_model import JointModel
from siamese.utils.training_utils import (
    get_callbacks, 
    plot_training_history, 
    evaluate_classification_model,
    enable_mixed_precision,
    set_random_seed
)

def train_separate_models(config=None, extractor_type='hybrid'):
    """
    Trains the Siamese Network and Classification Model separately.
    
    Parameters:
        config: Configuration object
        extractor_type: Type of feature extractor to use
        
    Returns:
        Trained models and evaluation metrics
    """
    if config is None:
        config = Config()
    
    # Set random seed
    set_random_seed(config=config)
    
    # Enable mixed precision if configured
    enable_mixed_precision(config=config)
    
    print("\n*** Starting model training ***\n")
    
    # Create data manager
    data_manager = DataManager(config=config)
    
    # Visualize dataset distribution
    print("Visualizing dataset distribution...")
    distribution = data_manager.visualize_dataset_distribution()
    print("Dataset distribution:")
    for split, counts in distribution.items():
        print(f"  {split}: {counts}")
    
    # Prepare siamese data for training and validation
    print("\nPreparing training data for Siamese Network...")
    X_train, y_train = data_manager.prepare_siamese_data(config.TRAIN_DIR, batch_size=config.BATCH_SIZE*10)
    
    print("\nPreparing validation data for Siamese Network...")
    X_val, y_val = data_manager.prepare_siamese_data(config.VAL_DIR, batch_size=config.BATCH_SIZE*5)
    
    # Check if data preparation was successful
    if X_train is None or X_val is None:
        print("Error: Could not prepare siamese training data. Check your dataset structure.")
        return None, None, None, None
    
    # Create Siamese Network
    print("\nCreating Siamese Network...")
    siamese_network = SiameseNetwork(config=config, extractor_type=extractor_type)
    
    # Train the Siamese network
    print("\nTraining Siamese Network...")
    siamese_callbacks = get_callbacks("siamese_model", config=config)
    siamese_history = siamese_network.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        callbacks=siamese_callbacks
    )
    
    # Plot training history
    plot_training_history(siamese_history, "Siamese_Network", config=config)
    
    # Get feature extractor from Siamese Network
    feature_extractor = siamese_network.get_feature_extractor()
    
    # Create data generators for classification
    train_gen, val_gen = data_manager.create_data_generators()
    
    # Create classification model with the pre-trained feature extractor
    print("\nCreating Classification Model...")
    classification_model = ClassificationModel(
        config=config, 
        feature_extractor=feature_extractor
    )
    
    # Train the classification model
    print("\nTraining Classification Model...")
    classification_callbacks = get_callbacks("classification_model", config=config)
    classification_history = classification_model.train(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS,
        steps_per_epoch=train_gen.samples // config.BATCH_SIZE,
        validation_steps=val_gen.samples // config.BATCH_SIZE,
        callbacks=classification_callbacks,
        unfreeze_after=config.EPOCHS // 3  # Unfreeze after 1/3 of total epochs
    )
    
    # Plot training history
    plot_training_history(classification_history, "Classification_Model", config=config)
    
    # Evaluate the classification model
    print("\nEvaluating Classification Model...")
    metrics = evaluate_classification_model(
        classification_model.model, 
        val_gen, 
        class_names=config.CLASS_NAMES, 
        config=config
    )
    
    # Print metrics
    print("\nClassification Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Save models
    print("\nSaving models...")
    siamese_network.save()
    classification_model.save()
    
    # Save feature extractor separately
    feature_extractor_path = os.path.join(config.CHECKPOINT_DIR, 'feature_extractor.keras')
    os.makedirs(os.path.dirname(feature_extractor_path), exist_ok=True)
    feature_extractor.save(feature_extractor_path)
    
    print("\nTraining and evaluation completed!")
    
    return siamese_network, classification_model, feature_extractor, metrics

def train_joint_model(config=None, extractor_type='hybrid'):
    """
    Trains the Siamese Network and Classification Model jointly.
    
    Parameters:
        config: Configuration object
        extractor_type: Type of feature extractor to use
        
    Returns:
        Trained joint model and evaluation metrics
    """
    if config is None:
        config = Config()
    
    # Set random seed
    set_random_seed(config=config)
    
    # Enable mixed precision if configured
    enable_mixed_precision(config=config)
    
    print("\n*** Starting joint model training ***\n")
    
    # Create data manager
    data_manager = DataManager(config=config)
    
    # Visualize dataset distribution
    print("Visualizing dataset distribution...")
    distribution = data_manager.visualize_dataset_distribution()
    print("Dataset distribution:")
    for split, counts in distribution.items():
        print(f"  {split}: {counts}")
    
    # Prepare siamese data for training and validation
    print("\nPreparing training data for Siamese Network...")
    X_train, y_train = data_manager.prepare_siamese_data(config.TRAIN_DIR, batch_size=config.BATCH_SIZE*10)
    
    print("\nPreparing validation data for Siamese Network...")
    X_val, y_val = data_manager.prepare_siamese_data(config.VAL_DIR, batch_size=config.BATCH_SIZE*5)
    
    # Check if data preparation was successful
    if X_train is None or X_val is None:
        print("Error: Could not prepare siamese training data. Check your dataset structure.")
        return None, None
    
    # Create data generators for classification
    train_gen, val_gen = data_manager.create_data_generators()
    
    # Create joint model
    print("\nCreating Joint Model...")
    joint_model = JointModel(config=config, extractor_type=extractor_type)
    
    # Get callbacks
    callbacks = get_callbacks("joint_model", config=config)
    
    # Train the joint model
    print("\nTraining Joint Model...")
    joint_history = joint_model.joint_train(
        siamese_train_data=(X_train, y_train),
        siamese_val_data=(X_val, y_val),
        class_train_data=train_gen,
        class_val_data=val_gen,
        epochs=config.EPOCHS,
        siamese_weight=0.5,  # Equal weight to both tasks
        callbacks=callbacks,
        unfreeze_after=config.EPOCHS // 3  # Unfreeze after 1/3 of total epochs
    )
    
    # Plot training history
    plot_training_history(joint_history, "Joint_Model", config=config)
    
    # Evaluate the classification model
    print("\nEvaluating Classification Model...")
    metrics = evaluate_classification_model(
        joint_model.classification_model.model, 
        val_gen, 
        class_names=config.CLASS_NAMES, 
        config=config
    )
    
    # Print metrics
    print("\nClassification Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Save models
    print("\nSaving models...")
    joint_model.save_models(
        siamese_path=os.path.join(config.CHECKPOINT_DIR, 'siamese_model_joint.keras'),
        classification_path=os.path.join(config.CHECKPOINT_DIR, 'classification_model_joint.keras'),
        feature_extractor_path=os.path.join(config.CHECKPOINT_DIR, 'feature_extractor_joint.keras')
    )
    
    print("\nJoint training and evaluation completed!")
    
    return joint_model, metrics

def main():
    """
    Main function to parse arguments and train models.
    """
    parser = argparse.ArgumentParser(description='Train OPG age classification models.')
    parser.add_argument('--mode', type=str, default='joint', choices=['separate', 'joint'],
                       help='Training mode: "separate" for training models separately, "joint" for joint training')
    parser.add_argument('--extractor', type=str, default='hybrid', choices=['cnn', 'vit', 'hybrid'],
                       help='Type of feature extractor to use')
    parser.add_argument('--train_dir', type=str, default=None,
                       help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default=None,
                       help='Directory containing validation data')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate for training')
    
    args = parser.parse_args()
    
    # Create a custom config with provided arguments
    config = Config()
    
    if args.train_dir:
        config.TRAIN_DIR = args.train_dir
    if args.val_dir:
        config.VAL_DIR = args.val_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    
    # Train models based on the specified mode
    if args.mode == 'separate':
        train_separate_models(config=config, extractor_type=args.extractor)
    else:
        train_joint_model(config=config, extractor_type=args.extractor)

if __name__ == "__main__":
    main() 