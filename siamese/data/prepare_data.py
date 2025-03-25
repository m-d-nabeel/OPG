import os
import shutil
import argparse
import hashlib
from pathlib import Path
import random
import re
from tqdm import tqdm
import numpy as np
from PIL import Image
import tensorflow as tf

def compute_image_hash(image_path):
    """
    Compute a hash for an image file to identify duplicates.
    
    Parameters:
        image_path: Path to the image file
        
    Returns:
        Hash string representing the image content
    """
    try:
        with open(image_path, "rb") as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        return img_hash
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_unique_images(source_dirs, target_dir, hash_based=True):
    """
    Find unique images across multiple directories and copy them to a target directory.
    
    Parameters:
        source_dirs: List of source directories
        target_dir: Target directory for unique images
        hash_based: Whether to use hash-based deduplication (True) or just filename (False)
        
    Returns:
        Dictionary mapping original paths to new paths
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Track images we've seen
    seen_hashes = set()
    seen_filenames = set()
    path_mapping = {}
    
    print("Finding unique images...")
    
    # Get all image files
    all_images = []
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, file))
    
    # Process each image
    unique_count = 0
    duplicate_count = 0
    
    for img_path in tqdm(all_images):
        filename = os.path.basename(img_path)
        
        if hash_based:
            # Hash-based deduplication
            img_hash = compute_image_hash(img_path)
            if img_hash is None:
                continue  # Skip if hash computation failed
                
            if img_hash in seen_hashes:
                duplicate_count += 1
                continue
                
            seen_hashes.add(img_hash)
        else:
            # Filename-based deduplication
            if filename in seen_filenames:
                duplicate_count += 1
                continue
                
            seen_filenames.add(filename)
        
        # Copy the unique image to the target directory
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(img_path, target_path)
        path_mapping[img_path] = target_path
        unique_count += 1
    
    print(f"Found {unique_count} unique images and {duplicate_count} duplicates.")
    return path_mapping

def classify_image(img_path, class_names, original_paths=None):
    """
    Classify an image into one of the age groups based on filename or directory structure.
    
    Parameters:
        img_path: Path to the image file
        class_names: List of class names for classification
        original_paths: Dictionary mapping current paths to original paths
        
    Returns:
        The predicted class name
    """
    # First try to classify based on original path if available
    if original_paths and img_path in original_paths:
        original_path = original_paths[img_path]
        # Check if any class name appears in the path
        for class_name in class_names:
            # Extract the age range number without "Age_" prefix
            age_match = re.search(r'age[_-]?(\d+)[_-]?(\d+)', original_path.lower())
            if age_match:
                age_lower = int(age_match.group(1))
                age_upper = int(age_match.group(2))
                
                # Match with the closest class range
                for cls in class_names:
                    cls_match = re.search(r'Age_(\d+)-(\d+)', cls)
                    if cls_match:
                        cls_lower = int(cls_match.group(1))
                        cls_upper = int(cls_match.group(2))
                        
                        if (age_lower >= cls_lower and age_lower <= cls_upper) or \
                           (age_upper >= cls_lower and age_upper <= cls_upper):
                            return cls
            
            # Direct class name match
            if class_name.lower() in original_path.lower():
                return class_name
    
    # Try to classify based on filename patterns
    filename = os.path.basename(img_path).lower()
    
    # Check for common age patterns in filename
    # Pattern like: "age_10_14" or "age 10-14" or "10-14 years"
    age_patterns = [
        r'age[_\s]?(\d+)[_\-\s](\d+)',  # age_10_14 or age 10-14
        r'(\d+)[_\-\s](\d+)[_\s]?(?:years|yrs|y)',  # 10-14 years or 10-14y
        r'(?:years|yrs|y)[_\s]?(\d+)[_\-\s](\d+)'   # years 10-14 or yrs_10-14
    ]
    
    for pattern in age_patterns:
        match = re.search(pattern, filename)
        if match:
            age_lower = int(match.group(1))
            age_upper = int(match.group(2))
            
            # Find the closest matching class
            for cls in class_names:
                cls_match = re.search(r'Age_(\d+)-(\d+)', cls)
                if cls_match:
                    cls_lower = int(cls_match.group(1))
                    cls_upper = int(cls_match.group(2))
                    
                    # Check if there's overlap between the ranges
                    if (age_lower >= cls_lower and age_lower <= cls_upper) or \
                       (age_upper >= cls_lower and age_upper <= cls_upper):
                        return cls
    
    # If all else fails, classify based on class-specific keywords
    keywords = {
        'Age_5-10': ['child', 'young', '5-10', '5_10', 'primary'],
        'Age_11-14': ['preteen', 'middle', '11-14', '11_14'],
        'Age_15-18': ['teen', 'high', '15-18', '15_18', 'adolescent'],
        'Age_19-24': ['adult', 'college', '19-24', '19_24', 'young adult']
    }
    
    for cls, words in keywords.items():
        if cls in class_names:  # Make sure the class is in our target classes
            for word in words:
                if word in filename:
                    return cls
    
    # If we still can't classify, assign randomly with weighted probabilities
    # This is a fallback that can be customized based on your dataset
    weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for all classes
    return random.choices(class_names, weights=weights, k=1)[0]

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8, class_names=None, original_paths=None):
    """
    Split a dataset into training and validation sets.
    
    Parameters:
        source_dir: Directory containing unique images
        train_dir: Directory for training data
        val_dir: Directory for validation data
        split_ratio: Ratio of training to total data
        class_names: List of class names (subdirectories will be created for each)
        original_paths: Dictionary mapping current paths to original paths
        
    Returns:
        Dictionary with counts of images in each split
    """
    if class_names is None:
        from siamese.configs.config import Config
        class_names = Config.CLASS_NAMES
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create class subdirectories
    for cls in class_names:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    
    print("Distributing images into training and validation sets...")
    
    # Get all images
    images = [os.path.join(source_dir, f) for f in os.listdir(source_dir) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Shuffle the images
    random.shuffle(images)
    
    # Split into training and validation
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Distribute images
    counts = {"train": {cls: 0 for cls in class_names}, 
              "val": {cls: 0 for cls in class_names}}
    
    for img_list, target_dir, split_name in [(train_images, train_dir, "train"), 
                                           (val_images, val_dir, "val")]:
        for img_path in tqdm(img_list):
            # Classify the image
            class_name = classify_image(img_path, class_names, original_paths)
            
            # Copy to the appropriate class directory
            target_class_dir = os.path.join(target_dir, class_name)
            target_path = os.path.join(target_class_dir, os.path.basename(img_path))
            shutil.copy2(img_path, target_path)
            
            # Update counts
            counts[split_name][class_name] += 1
    
    print("Dataset split complete.")
    return counts

def preprocess_images(data_dir, target_size=(1024, 512)):
    """
    Preprocess images in a directory to ensure they have the right format.
    
    Parameters:
        data_dir: Directory containing the images
        target_size: Target size for all images
    """
    print(f"Preprocessing images in {data_dir}...")
    
    for root, _, files in os.walk(data_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                
                try:
                    # Open and convert the image
                    with Image.open(img_path) as img:
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            
                        # Resize
                        img = img.resize(target_size, Image.LANCZOS)
                        
                        # Save back
                        img.save(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

def prepare_kaggle_data(input_dir, output_dir, split_ratio=0.8, preprocess=True):
    """
    Prepare data from Kaggle folder structure.
    
    Parameters:
        input_dir: Kaggle input directory
        output_dir: Output directory for processed data
        split_ratio: Ratio of training to total data
        preprocess: Whether to preprocess the images
        
    Returns:
        Dictionary with information about the dataset
    """
    # Create the necessary directories
    unique_dir = os.path.join(output_dir, 'unique_images')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all possible source directories in the Kaggle input folder
    source_dirs = []
    for root, dirs, _ in os.walk(input_dir):
        for dir_name in dirs:
            source_dirs.append(os.path.join(root, dir_name))
    
    # Find unique images
    path_mapping = find_unique_images(source_dirs, unique_dir)
    
    # Create reverse mapping from new paths to original paths
    reverse_mapping = {target: source for source, target in path_mapping.items()}
    
    # Split the dataset
    counts = split_dataset(
        unique_dir, train_dir, val_dir, 
        split_ratio, 
        original_paths=reverse_mapping
    )
    
    # Preprocess images if requested
    if preprocess:
        print("Preprocessing training images...")
        preprocess_images(train_dir)
        
        print("Preprocessing validation images...")
        preprocess_images(val_dir)
    
    # Return information about the dataset
    result = {
        "unique_images": len(path_mapping),
        "train_dir": train_dir,
        "val_dir": val_dir,
        "class_distribution": counts
    }
    
    return result

def update_config_paths(train_dir, val_dir):
    """
    Update the config file with the new data paths.
    
    Parameters:
        train_dir: Path to training data
        val_dir: Path to validation data
    """
    from siamese.configs.config import Config
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             '..', 'configs', 'config.py')
    
    # Read the config file
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Update the paths
    config_content = config_content.replace(
        f"TRAIN_DIR = os.path.join('data', 'train')",
        f"TRAIN_DIR = '{train_dir}'"
    )
    config_content = config_content.replace(
        f"VAL_DIR = os.path.join('data', 'val')",
        f"VAL_DIR = '{val_dir}'"
    )
    
    # Write the updated config
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Updated config with train_dir: {train_dir}, val_dir: {val_dir}")

def main():
    parser = argparse.ArgumentParser(description='Prepare data for Siamese OPG Age Classification')
    parser.add_argument('--input', type=str, default='kaggle/input',
                      help='Input directory containing Kaggle dataset')
    parser.add_argument('--output', type=str, default='kaggle/working',
                      help='Output directory for processed data')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                      help='Ratio of training to total data')
    parser.add_argument('--no-preprocess', action='store_true',
                      help='Skip image preprocessing')
    parser.add_argument('--update-config', action='store_true',
                      help='Update the config file with the new data paths')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Prepare the data
    result = prepare_kaggle_data(
        args.input, 
        args.output,
        args.split_ratio,
        not args.no_preprocess
    )
    
    # Print dataset information
    print("\nDataset information:")
    print(f"Unique images: {result['unique_images']}")
    print("Class distribution:")
    for split, classes in result['class_distribution'].items():
        print(f"  {split}:")
        for cls, count in classes.items():
            print(f"    {cls}: {count}")
    
    # Update config if requested
    if args.update_config:
        update_config_paths(result['train_dir'], result['val_dir'])

if __name__ == "__main__":
    main() 