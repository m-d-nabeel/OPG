import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from siamese.configs.config import Config
from siamese.models.feature_extractors import FeatureExtractorFactory

class ClassificationModel:
    def __init__(self, config=None, feature_extractor=None, extractor_type='hybrid', num_classes=None):
        """
        Initializes a Classification Model.
        
        Parameters:
            config: Configuration object
            feature_extractor: Pre-trained feature extractor from Siamese Network
            extractor_type: Type of feature extractor to create if one is not provided
            num_classes: Number of classes to classify
        """
        self.config = config or Config()
        self.input_shape = (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3)
        self.num_classes = num_classes or self.config.NUM_CLASSES
        
        # Create or use the feature extractor
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractorFactory.create_feature_extractor(
                extractor_type=extractor_type,
                input_shape=self.input_shape,
                config=self.config
            )
        else:
            self.feature_extractor = feature_extractor
            
        # Set feature extractor as non-trainable initially
        # This will be changed during the training process
        for layer in self.feature_extractor.layers:
            layer.trainable = False
        
        # Build the classification model
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Builds and returns the Classification Model.
        """
        # Create an input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Get features using the feature extractor
        features = self.feature_extractor(inputs)
        
        # Add classification layers with attention mechanism
        x = layers.Dense(256, activation='relu')(features)
        x = layers.BatchNormalization()(x)
        
        # Self-attention layer
        attention = layers.Dense(256, activation='tanh')(x)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        attention_weights = layers.Multiply()([x, attention])
        x = layers.Add()([x, attention_weights])
        
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs, name="classification_model")
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            metrics=['accuracy']
        )
        
        return model
    
    def unfreeze_feature_extractor(self, fine_tuning_lr=None):
        """
        Unfreezes the feature extractor for fine-tuning.
        
        Parameters:
            fine_tuning_lr: Learning rate for fine-tuning
        """
        # Make feature extractor trainable
        for layer in self.feature_extractor.layers:
            layer.trainable = True
        
        # Recompile with a lower learning rate for fine-tuning
        if fine_tuning_lr is None:
            fine_tuning_lr = self.config.LEARNING_RATE / 10.0
            
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizers.Adam(learning_rate=fine_tuning_lr),
            metrics=['accuracy']
        )
    
    def train(self, train_data, validation_data=None, epochs=None, steps_per_epoch=None, validation_steps=None, callbacks=None, unfreeze_after=10):
        """
        Trains the Classification Model with option to unfreeze the feature extractor.
        
        Parameters:
            train_data: Training data
            validation_data: Validation data
            epochs: Number of epochs to train for
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps
            callbacks: List of callbacks for training
            unfreeze_after: Number of epochs after which to unfreeze the feature extractor
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = self.config.EPOCHS
        
        # Initial training with frozen feature extractor
        print("Training with frozen feature extractor...")
        initial_history = self.model.fit(
            train_data,
            epochs=unfreeze_after if unfreeze_after < epochs else epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        # If we still have epochs left and should unfreeze
        if epochs > unfreeze_after:
            # Unfreeze feature extractor for fine-tuning
            print("Unfreezing feature extractor for fine-tuning...")
            self.unfreeze_feature_extractor()
            
            # Continue training with unfrozen feature extractor
            fine_tuning_history = self.model.fit(
                train_data,
                initial_epoch=unfreeze_after,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data,
                validation_steps=validation_steps,
                callbacks=callbacks
            )
            
            # Combine histories
            combined_history = {}
            for key in initial_history.history:
                combined_history[key] = initial_history.history[key] + fine_tuning_history.history[key]
                
            # Create a new history object with the combined data
            class CombinedHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
                    
            return CombinedHistory(combined_history)
        
        return initial_history
    
    def evaluate(self, test_data, steps=None):
        """
        Evaluates the model on test data.
        
        Parameters:
            test_data: Test data
            steps: Number of steps for evaluation
            
        Returns:
            Evaluation metrics
        """
        return self.model.evaluate(test_data, steps=steps)
    
    def predict(self, x):
        """
        Makes predictions on new data.
        
        Parameters:
            x: Input data
            
        Returns:
            Predictions
        """
        return self.model.predict(x)
    
    def save(self, path=None):
        """
        Saves the classification model.
        
        Parameters:
            path: Path to save the model to
        """
        if path is None:
            path = self.config.CLASSIFICATION_MODEL_PATH
            
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        
    @classmethod
    def load(cls, path=None, config=None, feature_extractor=None):
        """
        Loads a classification model from disk.
        
        Parameters:
            path: Path to load the model from
            config: Configuration object
            feature_extractor: Feature extractor to use
            
        Returns:
            A ClassificationModel instance
        """
        if config is None:
            config = Config()
            
        if path is None:
            path = config.CLASSIFICATION_MODEL_PATH
        
        # Load the model
        model = tf.keras.models.load_model(path)
        
        # Create a new instance
        instance = cls(config=config, feature_extractor=feature_extractor)
        instance.model = model
        
        return instance 