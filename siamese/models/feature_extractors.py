import tensorflow as tf
from tensorflow.keras import layers, models, applications
from siamese.configs.config import Config

# Multi-layer perception mixer for ViT
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# Patch creation and embedding for ViT
def create_patches(images, patch_size):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches

# Vision Transformer Encoder Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class FeatureExtractorFactory:
    """Factory class for creating different types of feature extractors"""
    
    @staticmethod
    def create_feature_extractor(extractor_type='hybrid', input_shape=None, config=None):
        """
        Creates and returns the specified type of feature extractor.
        
        Parameters:
            extractor_type: Type of extractor ('cnn', 'vit', or 'hybrid')
            input_shape: Shape of the input images
            config: Configuration object
            
        Returns:
            A feature extractor model
        """
        if config is None:
            config = Config()
            
        if input_shape is None:
            input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, 3)
        
        if extractor_type.lower() == 'cnn':
            return FeatureExtractorFactory.create_cnn_extractor(input_shape, config)
        elif extractor_type.lower() == 'vit':
            return FeatureExtractorFactory.create_vit_extractor(input_shape, config)
        elif extractor_type.lower() == 'hybrid':
            return FeatureExtractorFactory.create_hybrid_extractor(input_shape, config)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    @staticmethod
    def create_cnn_extractor(input_shape, config):
        """Creates a CNN-based feature extractor"""
        # Use a pre-trained CNN model as backbone
        base_model = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet'
        )
        
        # Create the model
        inputs = layers.Input(shape=input_shape)
        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(config.FEATURE_DIMENSION, activation='relu')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name="cnn_feature_extractor")
        return model
    
    @staticmethod
    def create_vit_extractor(input_shape, config):
        """Creates a Vision Transformer feature extractor"""
        # Calculate ViT parameters
        patch_size = config.PATCH_SIZE
        num_patches = config.NUM_PATCHES
        projection_dim = config.PROJECTION_DIM
        num_heads = config.NUM_HEADS
        transformer_layers = config.TRANSFORMER_LAYERS
        mlp_head_units = config.MLP_HEAD_UNITS
        
        inputs = layers.Input(shape=input_shape)
        
        # Create patches and project them
        patches = layers.Lambda(
            lambda x: create_patches(x, patch_size)
        )(inputs)
        
        # Add positional embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)
        
        # Project patches to projection_dim
        patch_embedding = layers.Dense(projection_dim)(patches)
        x = patch_embedding + position_embedding
        
        # Add dropout
        x = layers.Dropout(0.1)(x)
        
        # Create transformer blocks
        for _ in range(transformer_layers):
            x = TransformerBlock(projection_dim, num_heads, projection_dim*2)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Create MLP head
        for dim in mlp_head_units:
            x = layers.Dense(dim, activation="gelu")(x)
            x = layers.Dropout(0.1)(x)
        
        # Final output layer
        outputs = layers.Dense(config.FEATURE_DIMENSION)(x)
        
        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs, name="vit_feature_extractor")
        return model
    
    @staticmethod
    def create_hybrid_extractor(input_shape, config):
        """Creates a hybrid feature extractor with both CNN and ViT features"""
        # Use a pre-trained CNN model as one backbone
        cnn_base_model = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False, 
            weights='imagenet'
        )
        
        # Create ViT model as second backbone
        vit_model = FeatureExtractorFactory.create_vit_extractor(input_shape, config)
        
        # Define the input layer
        inputs = layers.Input(shape=input_shape)
        
        # Get features from CNN backbone
        cnn_features = cnn_base_model(inputs)
        cnn_features = layers.GlobalAveragePooling2D()(cnn_features)
        cnn_features = layers.Dense(config.FEATURE_DIMENSION, activation='relu')(cnn_features)
        
        # Get features from ViT backbone
        vit_features = vit_model(inputs)
        
        # Concatenate features from both backbones
        combined_features = layers.Concatenate()([cnn_features, vit_features])
        
        # Add final layers for feature extraction
        x = layers.Dense(512, activation='relu')(combined_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(config.FEATURE_DIMENSION, activation='relu')(x)
        
        # Create and return the model
        model = models.Model(inputs=inputs, outputs=outputs, name="hybrid_feature_extractor")
        return model 