import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from itertools import combinations
from siamese.configs.config import Config

class DataManager:
    def __init__(self, config=None):
        self.config = config or Config()
        
    def create_data_generators(self):
        """
        Creates and returns data generators for training and validation.
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            brightness_range=self.config.BRIGHTNESS_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP,
            fill_mode=self.config.FILL_MODE,
            cval=self.config.CVAL
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            self.config.TRAIN_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )
        
        val_gen = val_datagen.flow_from_directory(
            self.config.VAL_DIR,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, val_gen
    
    def prepare_siamese_data(self, directory, batch_size=None):
        """
        Prepares data for Siamese Network training by creating pairs and labels.
        Returns arrays of image pairs and their labels (1 for same class, 0 for different).
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        # Create standard data generator for loading images
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=self.config.ROTATION_RANGE,
            width_shift_range=self.config.WIDTH_SHIFT_RANGE,
            height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
            brightness_range=self.config.BRIGHTNESS_RANGE,
            zoom_range=self.config.ZOOM_RANGE,
            horizontal_flip=self.config.HORIZONTAL_FLIP,
            fill_mode=self.config.FILL_MODE,
            cval=self.config.CVAL
        )
        
        # Load all images by class
        class_images = {}
        for class_name in self.config.CLASS_NAMES:
            class_dir = os.path.join(directory, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Get list of image files
            image_files = [f for f in os.listdir(class_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # If there are images in this class
            if image_files:
                class_images[class_name] = []
                for img_file in image_files:
                    img_path = os.path.join(class_dir, img_file)
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, 
                        target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    class_images[class_name].append(img_array)
        
        # Pairs and labels lists
        pairs = []
        labels = []
        
        # Create positive pairs (same class)
        for class_name, images in class_images.items():
            if len(images) < 2:
                continue
                
            # Create positive pairs using combinations
            positive_pairs = list(combinations(range(len(images)), 2))
            np.random.shuffle(positive_pairs)
            
            for idx1, idx2 in positive_pairs[:batch_size//2]:  # Limit to half the batch size
                pairs.append([images[idx1], images[idx2]])
                labels.append(1)  # Same class
                
                if len(pairs) >= batch_size//2:
                    break
        
        # Create negative pairs (different classes)
        class_names = list(class_images.keys())
        if len(class_names) >= 2:
            class_combinations = list(combinations(class_names, 2))
            np.random.shuffle(class_combinations)
            
            for class1, class2 in class_combinations:
                if len(pairs) >= batch_size:
                    break
                    
                images1 = class_images[class1]
                images2 = class_images[class2]
                
                if not images1 or not images2:
                    continue
                    
                for _ in range(min(len(images1), len(images2), batch_size - len(pairs))):
                    idx1 = np.random.randint(0, len(images1))
                    idx2 = np.random.randint(0, len(images2))
                    
                    pairs.append([images1[idx1], images2[idx2]])
                    labels.append(0)  # Different classes
        
        # Ensure we have pairs
        if not pairs:
            return None, None
            
        # Shuffle pairs and labels
        temp = list(zip(pairs, labels))
        np.random.shuffle(temp)
        pairs, labels = zip(*temp)
        
        # Convert to numpy arrays in the right format
        x1 = np.array([pair[0] for pair in pairs])
        x2 = np.array([pair[1] for pair in pairs])
        labels = np.array(labels)
        
        return [x1, x2], labels

    def create_tf_dataset(self, directory, is_training=True, is_siamese=False):
        """
        Creates a TensorFlow dataset that can be used for model.fit
        
        Parameters:
            directory: Directory containing class subdirectories
            is_training: Whether to apply data augmentation
            is_siamese: Whether to prepare siamese pairs
            
        Returns:
            A TensorFlow dataset
        """
        if is_siamese:
            X, y = self.prepare_siamese_data(directory, batch_size=self.config.BATCH_SIZE*10)
            if X is None:
                return None
            
            # Create a TensorFlow dataset from the prepared data
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            dataset = dataset.batch(self.config.BATCH_SIZE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
        else:
            # Create standard data generator
            if is_training:
                datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=self.config.ROTATION_RANGE,
                    width_shift_range=self.config.WIDTH_SHIFT_RANGE,
                    height_shift_range=self.config.HEIGHT_SHIFT_RANGE,
                    brightness_range=self.config.BRIGHTNESS_RANGE,
                    zoom_range=self.config.ZOOM_RANGE,
                    horizontal_flip=self.config.HORIZONTAL_FLIP,
                    fill_mode=self.config.FILL_MODE,
                    cval=self.config.CVAL
                )
            else:
                datagen = ImageDataGenerator(rescale=1./255)
                
            # Flow from directory
            generator = datagen.flow_from_directory(
                directory,
                target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                batch_size=self.config.BATCH_SIZE,
                class_mode='categorical',
                shuffle=is_training
            )
            
            # Convert to TF dataset
            dataset = tf.data.Dataset.from_generator(
                lambda: generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, self.config.NUM_CLASSES), dtype=tf.float32)
                )
            )
            
            return dataset
            
    def visualize_dataset_distribution(self):
        """
        Visualizes the distribution of images across different classes
        in the training and validation sets.
        """
        import matplotlib.pyplot as plt
        
        # Count images in each class
        train_counts = []
        val_counts = []
        
        for class_name in self.config.CLASS_NAMES:
            train_class_dir = os.path.join(self.config.TRAIN_DIR, class_name)
            val_class_dir = os.path.join(self.config.VAL_DIR, class_name)
            
            if os.path.isdir(train_class_dir):
                train_count = len([f for f in os.listdir(train_class_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
                train_counts.append(train_count)
            else:
                train_counts.append(0)
            
            if os.path.isdir(val_class_dir):
                val_count = len([f for f in os.listdir(val_class_dir) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))])
                val_counts.append(val_count)
            else:
                val_counts.append(0)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.DATASET_DISTRIBUTION_PATH), exist_ok=True)
        
        # Plot the distribution
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(self.config.CLASS_NAMES))
        width = 0.35
        
        plt.bar(x - width/2, train_counts, width, label='Training')
        plt.bar(x + width/2, val_counts, width, label='Validation')
        
        plt.xlabel('Age Ranges')
        plt.ylabel('Number of Images')
        plt.title('Dataset Distribution')
        plt.xticks(x, self.config.CLASS_NAMES)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.config.DATASET_DISTRIBUTION_PATH)
        plt.close()
        
        # Return the counts
        return {
            'train': dict(zip(self.config.CLASS_NAMES, train_counts)),
            'validation': dict(zip(self.config.CLASS_NAMES, val_counts))
        } 