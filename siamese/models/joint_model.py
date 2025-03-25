import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from siamese.configs.config import Config
from siamese.models.feature_extractors import FeatureExtractorFactory
from siamese.models.siamese_model import SiameseNetwork
from siamese.models.classification_model import ClassificationModel

class JointModel:
    def __init__(self, config=None, feature_extractor=None, extractor_type='hybrid'):
        """
        Initializes a Joint Model that combines Siamese Network and Classification Model.
        
        Parameters:
            config: Configuration object
            feature_extractor: Feature extractor to use (if None, one will be created)
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
        
        # Create the Siamese Network and Classification Model
        self.siamese_network = SiameseNetwork(
            config=self.config, 
            feature_extractor=self.feature_extractor
        )
        
        self.classification_model = ClassificationModel(
            config=self.config, 
            feature_extractor=self.feature_extractor
        )
        
        # Build the joint model
        self.joint_model = self._build_joint_model()
    
    def _build_joint_model(self):
        """
        Builds and returns a custom model for joint training.
        This is primarily a custom training loop since TF doesn't easily support
        models with different inputs training jointly.
        """
        # We don't actually build a joint model here, as we'll use a custom training loop
        # instead of a single model. This is because the Siamese network and classification 
        # model have different input structures.
        return None
    
    def joint_train(self, 
                   siamese_train_data, siamese_val_data,
                   class_train_data, class_val_data,
                   epochs=None, 
                   siamese_weight=0.5,
                   callbacks=None,
                   unfreeze_after=10):
        """
        Trains both the Siamese Network and Classification Model together.
        
        Parameters:
            siamese_train_data: Training data for Siamese Network (X, y)
            siamese_val_data: Validation data for Siamese Network (X, y)
            class_train_data: Training data for Classification Model (generator or dataset)
            class_val_data: Validation data for Classification Model (generator or dataset)
            epochs: Number of epochs to train for
            siamese_weight: Weight for Siamese loss (1-siamese_weight for classification loss)
            callbacks: List of callbacks for training
            unfreeze_after: Number of epochs after which to unfreeze the feature extractor
            
        Returns:
            Training history for both models
        """
        if epochs is None:
            epochs = self.config.EPOCHS
            
        # Check if we're using generators or datasets
        if hasattr(class_train_data, 'steps_per_epoch'):
            steps_per_epoch = class_train_data.steps_per_epoch
        elif hasattr(class_train_data, 'cardinality'):
            steps_per_epoch = tf.data.experimental.cardinality(class_train_data).numpy()
        else:
            steps_per_epoch = len(class_train_data) // self.config.BATCH_SIZE
            
        # Unpack Siamese data
        X_train_siamese, y_train_siamese = siamese_train_data
        X_val_siamese, y_val_siamese = siamese_val_data
        
        # Dictionary to store training history
        history = {
            'siamese_loss': [], 'siamese_acc': [], 
            'siamese_val_loss': [], 'siamese_val_acc': [],
            'class_loss': [], 'class_acc': [], 
            'class_val_loss': [], 'class_val_acc': [],
            'joint_loss': [], 'joint_val_loss': []
        }
        
        # Initialize optimizers with the same learning rate
        optimizer = optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # If we reach unfreeze_after epoch, unfreeze the feature extractor
            if epoch == unfreeze_after:
                print("Unfreezing feature extractor...")
                for layer in self.feature_extractor.layers:
                    layer.trainable = True
                
                # Reduce learning rate for fine-tuning
                optimizer.learning_rate = self.config.LEARNING_RATE / 10.0
            
            # Initialize metrics
            siamese_epoch_loss = 0
            siamese_epoch_acc = 0
            class_epoch_loss = 0
            class_epoch_acc = 0
            joint_epoch_loss = 0
            
            # Create iterators for class_train_data
            if hasattr(class_train_data, 'as_numpy_iterator'):
                class_iterator = iter(class_train_data)
            else:
                # If it's a generator
                class_iterator = class_train_data
                
            # Train on batches
            for step in range(steps_per_epoch):
                # Get batch from Siamese data (randomly)
                batch_indices = tf.random.uniform(
                    shape=[self.config.BATCH_SIZE], 
                    maxval=len(y_train_siamese), 
                    dtype=tf.int32
                )
                
                batch_X_siamese = [X_train_siamese[0][batch_indices], X_train_siamese[1][batch_indices]]
                batch_y_siamese = y_train_siamese[batch_indices]
                
                # Get batch from classification data
                try:
                    if hasattr(class_iterator, '__next__'):
                        batch_X_class, batch_y_class = next(class_iterator)
                    else:
                        # If it's a generator
                        batch_X_class, batch_y_class = class_iterator.__next__()
                except (StopIteration, tf.errors.OutOfRangeError):
                    # Reset iterator if we run out of data
                    if hasattr(class_train_data, 'as_numpy_iterator'):
                        class_iterator = iter(class_train_data)
                        batch_X_class, batch_y_class = next(class_iterator)
                    else:
                        # Reset generator, this is approximate
                        class_iterator = class_train_data
                        batch_X_class, batch_y_class = class_iterator.__next__()
                
                # Joint training step with gradients
                with tf.GradientTape() as tape:
                    # Forward pass for Siamese
                    siamese_preds = self.siamese_network.model(batch_X_siamese)
                    siamese_loss = tf.keras.losses.binary_crossentropy(batch_y_siamese, siamese_preds)
                    siamese_loss = tf.reduce_mean(siamese_loss)
                    
                    # Forward pass for Classification
                    class_preds = self.classification_model.model(batch_X_class)
                    class_loss = tf.keras.losses.categorical_crossentropy(batch_y_class, class_preds)
                    class_loss = tf.reduce_mean(class_loss)
                    
                    # Weighted combined loss
                    joint_loss = siamese_weight * siamese_loss + (1 - siamese_weight) * class_loss
                
                # Calculate gradients and apply them
                trainable_vars = self.feature_extractor.trainable_variables + \
                                self.siamese_network.model.trainable_variables + \
                                self.classification_model.model.trainable_variables
                
                gradients = tape.gradient(joint_loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                # Calculate metrics
                siamese_acc = tf.keras.metrics.binary_accuracy(batch_y_siamese, siamese_preds)
                siamese_acc = tf.reduce_mean(siamese_acc)
                
                class_acc = tf.keras.metrics.categorical_accuracy(batch_y_class, class_preds)
                class_acc = tf.reduce_mean(class_acc)
                
                # Update epoch metrics
                siamese_epoch_loss += siamese_loss
                siamese_epoch_acc += siamese_acc
                class_epoch_loss += class_loss
                class_epoch_acc += class_acc
                joint_epoch_loss += joint_loss
                
                # Print progress
                if step % 10 == 0:
                    print(f"Step {step}/{steps_per_epoch} - joint_loss: {joint_loss:.4f} - siamese_loss: {siamese_loss:.4f} - class_loss: {class_loss:.4f}")
            
            # Average epoch metrics
            siamese_epoch_loss /= steps_per_epoch
            siamese_epoch_acc /= steps_per_epoch
            class_epoch_loss /= steps_per_epoch
            class_epoch_acc /= steps_per_epoch
            joint_epoch_loss /= steps_per_epoch
            
            # Evaluate on validation data
            # Siamese validation
            siamese_val_loss, siamese_val_acc = self.siamese_network.model.evaluate(
                X_val_siamese, y_val_siamese, verbose=0
            )
            
            # Classification validation
            class_val_loss, class_val_acc = self.classification_model.model.evaluate(
                class_val_data, verbose=0
            )
            
            # Calculate joint validation loss
            joint_val_loss = siamese_weight * siamese_val_loss + (1 - siamese_weight) * class_val_loss
            
            # Update history
            history['siamese_loss'].append(float(siamese_epoch_loss))
            history['siamese_acc'].append(float(siamese_epoch_acc))
            history['siamese_val_loss'].append(float(siamese_val_loss))
            history['siamese_val_acc'].append(float(siamese_val_acc))
            
            history['class_loss'].append(float(class_epoch_loss))
            history['class_acc'].append(float(class_epoch_acc))
            history['class_val_loss'].append(float(class_val_loss))
            history['class_val_acc'].append(float(class_val_acc))
            
            history['joint_loss'].append(float(joint_epoch_loss))
            history['joint_val_loss'].append(float(joint_val_loss))
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs} - joint_loss: {joint_epoch_loss:.4f} - joint_val_loss: {joint_val_loss:.4f}")
            print(f"siamese_loss: {siamese_epoch_loss:.4f} - siamese_val_loss: {siamese_val_loss:.4f} - siamese_acc: {siamese_epoch_acc:.4f} - siamese_val_acc: {siamese_val_acc:.4f}")
            print(f"class_loss: {class_epoch_loss:.4f} - class_val_loss: {class_val_loss:.4f} - class_acc: {class_epoch_acc:.4f} - class_val_acc: {class_val_acc:.4f}")
            
            # Execute callbacks if provided
            if callbacks:
                for callback in callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, {
                            'loss': joint_epoch_loss,
                            'val_loss': joint_val_loss,
                            'siamese_loss': siamese_epoch_loss,
                            'siamese_val_loss': siamese_val_loss,
                            'siamese_acc': siamese_epoch_acc,
                            'siamese_val_acc': siamese_val_acc,
                            'class_loss': class_epoch_loss,
                            'class_val_loss': class_val_loss,
                            'class_acc': class_epoch_acc,
                            'class_val_acc': class_val_acc
                        })
        
        # Return the training history
        class TrainingHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return TrainingHistory(history)
    
    def save_models(self, siamese_path=None, classification_path=None, feature_extractor_path=None):
        """
        Saves both models and the feature extractor.
        
        Parameters:
            siamese_path: Path to save the Siamese model to
            classification_path: Path to save the Classification model to
            feature_extractor_path: Path to save the Feature Extractor to
        """
        # Save the Siamese model
        self.siamese_network.save(path=siamese_path)
        
        # Save the Classification model
        self.classification_model.save(path=classification_path)
        
        # Save the Feature Extractor separately if a path is provided
        if feature_extractor_path is not None:
            import os
            os.makedirs(os.path.dirname(feature_extractor_path), exist_ok=True)
            self.feature_extractor.save(feature_extractor_path)
            
    @classmethod
    def load_models(cls, siamese_path=None, classification_path=None, feature_extractor_path=None, config=None):
        """
        Loads both models and the feature extractor.
        
        Parameters:
            siamese_path: Path to load the Siamese model from
            classification_path: Path to load the Classification model from
            feature_extractor_path: Path to load the Feature Extractor from
            config: Configuration object
            
        Returns:
            A JointModel instance
        """
        if config is None:
            config = Config()
            
        # Load the feature extractor
        if feature_extractor_path is not None:
            feature_extractor = tf.keras.models.load_model(feature_extractor_path)
        else:
            feature_extractor = None
            
        # Create a new instance
        instance = cls(config=config, feature_extractor=feature_extractor)
        
        # Load the Siamese model if a path is provided
        if siamese_path is not None:
            instance.siamese_network = SiameseNetwork.load(path=siamese_path, config=config)
            
        # Load the Classification model if a path is provided
        if classification_path is not None:
            instance.classification_model = ClassificationModel.load(
                path=classification_path, 
                config=config, 
                feature_extractor=instance.feature_extractor
            )
            
        return instance 