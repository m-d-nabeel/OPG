import os
import numpy as np
import tensorflow as tf
from siamese.configs.config import Config
from siamese.models.classification_model import ClassificationModel

class OPGPredictor:
    def __init__(self, model_path=None, config=None):
        """
        Initializes a predictor for OPG images.
        
        Parameters:
            model_path: Path to the saved model
            config: Configuration object
        """
        self.config = config or Config()
        
        # Load the model
        if model_path is None:
            model_path = self.config.CLASSIFICATION_MODEL_PATH
            
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Get class names
        self.class_names = self.config.CLASS_NAMES
    
    def predict_age_range(self, image_path):
        """
        Predicts the age range of a new OPG image.
        
        Parameters:
            image_path: Path to the image
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        prediction = self.model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Get prediction probabilities for each class
        probabilities = prediction[0]
        
        # Return the predicted class and probabilities
        result = {
            'predicted_age_range': self.class_names[predicted_class],
            'confidence': float(probabilities[predicted_class]),
            'class_probabilities': {self.class_names[i]: float(probabilities[i]) for i in range(len(self.class_names))}
        }
        
        return result
    
    def batch_predict(self, image_paths):
        """
        Makes predictions on a batch of images.
        
        Parameters:
            image_paths: List of paths to images
            
        Returns:
            List of prediction results
        """
        # Load and preprocess all images
        batch_img_arrays = []
        
        for image_path in image_paths:
            img = tf.keras.preprocessing.image.load_img(
                image_path, 
                target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            batch_img_arrays.append(img_array)
        
        # Convert to batch
        batch_img_arrays = np.array(batch_img_arrays) / 255.0
        
        # Make predictions
        predictions = self.model.predict(batch_img_arrays)
        
        # Process results
        results = []
        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction)
            
            result = {
                'image_path': image_paths[i],
                'predicted_age_range': self.class_names[predicted_class],
                'confidence': float(prediction[predicted_class]),
                'class_probabilities': {self.class_names[j]: float(prediction[j]) for j in range(len(self.class_names))}
            }
            
            results.append(result)
        
        return results

class SiameseImageComparer:
    def __init__(self, model_path=None, feature_extractor_path=None, config=None):
        """
        Initializes a comparer for OPG images using Siamese network.
        
        Parameters:
            model_path: Path to the saved Siamese model
            feature_extractor_path: Path to the feature extractor
            config: Configuration object
        """
        self.config = config or Config()
        
        # Load the feature extractor
        if feature_extractor_path is None:
            feature_extractor_path = os.path.join(
                os.path.dirname(self.config.SIAMESE_MODEL_PATH),
                'feature_extractor.keras'
            )
            
        if os.path.exists(feature_extractor_path):
            self.feature_extractor = tf.keras.models.load_model(feature_extractor_path)
        else:
            raise FileNotFoundError(f"Feature extractor not found at {feature_extractor_path}")
            
        # Load the Siamese model
        if model_path is None:
            model_path = self.config.SIAMESE_MODEL_PATH
            
        if os.path.exists(model_path):
            self.siamese_model = tf.keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Siamese model not found at {model_path}")
    
    def compare_images(self, image_path1, image_path2):
        """
        Compares two OPG images and returns their similarity.
        
        Parameters:
            image_path1: Path to the first image
            image_path2: Path to the second image
            
        Returns:
            Similarity score between 0 and 1
        """
        # Load and preprocess the images
        img1 = tf.keras.preprocessing.image.load_img(
            image_path1, 
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        )
        img1_array = tf.keras.preprocessing.image.img_to_array(img1)
        img1_array = np.expand_dims(img1_array, axis=0) / 255.0
        
        img2 = tf.keras.preprocessing.image.load_img(
            image_path2, 
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        )
        img2_array = tf.keras.preprocessing.image.img_to_array(img2)
        img2_array = np.expand_dims(img2_array, axis=0) / 255.0
        
        # Make prediction with Siamese model
        similarity = self.siamese_model.predict([img1_array, img2_array])[0][0]
        
        return float(similarity)
    
    def find_similar_images(self, query_image_path, dataset_dir, top_k=5):
        """
        Finds similar images to the query image in the dataset.
        
        Parameters:
            query_image_path: Path to the query image
            dataset_dir: Directory containing dataset images
            top_k: Number of similar images to return
            
        Returns:
            List of dictionaries with similar images and their similarity scores
        """
        # Load and preprocess the query image
        query_img = tf.keras.preprocessing.image.load_img(
            query_image_path, 
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
        )
        query_img_array = tf.keras.preprocessing.image.img_to_array(query_img)
        query_img_array = np.expand_dims(query_img_array, axis=0) / 255.0
        
        # Extract features from the query image
        query_features = self.feature_extractor.predict(query_img_array)[0]
        
        # Find all images in the dataset
        all_images = []
        image_paths = []
        
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    image_paths.append(img_path)
                    
                    # Load and preprocess the image
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, 
                        target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    
                    all_images.append(img_array)
        
        # Concatenate all images into a single batch
        if len(all_images) > 0:
            batch_images = np.vstack(all_images)
            
            # Extract features from all images
            batch_features = self.feature_extractor.predict(batch_images)
            
            # Calculate similarities (using Euclidean distance)
            similarities = []
            for i, features in enumerate(batch_features):
                distance = np.sqrt(np.sum(np.square(query_features - features)))
                similarities.append((image_paths[i], distance))
            
            # Sort by similarity (smaller distance = more similar)
            similarities.sort(key=lambda x: x[1])
            
            # Return top_k most similar images (excluding the query image itself)
            most_similar = []
            for path, distance in similarities[:top_k+1]:
                if path != query_image_path:  # Exclude the query image
                    class_name = os.path.basename(os.path.dirname(path))
                    most_similar.append({
                        'image_path': path,
                        'class': class_name,
                        'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to a similarity score
                    })
                    
                    if len(most_similar) >= top_k:
                        break
            
            return most_similar
        
        return [] 