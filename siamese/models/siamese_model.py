import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from siamese.configs.config import Config
from siamese.models.feature_extractors import FeatureExtractorFactory

class SiameseNetwork:
    def __init__(self, config=None, feature_extractor=None, extractor_type='hybrid'):
        """
        Initializes a Siamese Neural Network.
        
        Parameters:
            config: Configuration object
            feature_extractor: Pre-trained feature extractor (if None, one will be created)
            extractor_type: Type of feature extractor to create if one is not provided
        """
        self.config = config or Config()
        self.input_shape = (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3)
        
        # Create or use the feature extractor
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(
                extractor_type=extractor_type,
                input_shape=self.input_shape,
                config=self.config
            )
        else:
            self.feature_extractor = feature_extractor
        
        # Build the Siamese model
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Builds and returns the Siamese Neural Network model.
        """
        # Define the inputs for each image
        input_a = layers.Input(shape=self.input_shape)
        input_b = layers.Input(shape=self.input_shape)
        
        # Get feature vectors for both images using the same extractor
        feat_a = self.feature_extractor(input_a)
        feat_b = self.feature_extractor(input_b)
        
        # Calculate the absolute difference between the feature vectors
        distance = layers.Lambda(lambda x: K.abs(x[0] - x[1]))([feat_a, feat_b])
        
        # Add layers to interpret the distance
        x = layers.Dense(128, activation='relu')(distance)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        # Define the model with two inputs and one output
        model = models.Model(inputs=[input_a, input_b], outputs=output, name="siamese_network")
        
        # Compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            metrics=['accuracy']
        )
        
        return model
    
    def get_feature_extractor(self):
        """
        Returns the feature extractor used by the Siamese Network.
        """
        return self.feature_extractor
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None, callbacks=None):
        """
        Trains the Siamese Network.
        
        Parameters:
            X_train: Training data pairs
            y_train: Training labels
            X_val: Validation data pairs
            y_val: Validation labels
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            callbacks: List of callbacks for training
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.EPOCHS
            
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )
        
        return history
    
    def save(self, path=None):
        """
        Saves the Siamese model and feature extractor.
        
        Parameters:
            path: Path to save the model to
        """
        if path is None:
            path = self.config.SIAMESE_MODEL_PATH
            
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the Siamese model
        self.model.save(path)
        
        # Save the feature extractor
        feature_extractor_path = os.path.join(
            os.path.dirname(path),
            'feature_extractor.keras'
        )
        self.feature_extractor.save(feature_extractor_path)
        
    @classmethod
    def load(cls, path=None, config=None):
        """
        Loads a Siamese model from disk.
        
        Parameters:
            path: Path to load the model from
            config: Configuration object
            
        Returns:
            A SiameseNetwork instance
        """
        if config is None:
            config = Config()
            
        if path is None:
            path = config.SIAMESE_MODEL_PATH
            
        # Load the model
        model = tf.keras.models.load_model(path)
        
        # Load the feature extractor
        feature_extractor_path = os.path.join(
            os.path.dirname(path),
            'feature_extractor.keras'
        )
        feature_extractor = tf.keras.models.load_model(feature_extractor_path)
        
        # Create a new instance
        instance = cls(config=config, feature_extractor=feature_extractor)
        instance.model = model
        
        return instance 