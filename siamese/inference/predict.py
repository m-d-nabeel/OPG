import os
import argparse
import glob
from siamese.configs.config import Config
from siamese.inference.predictor import OPGPredictor, SiameseImageComparer

def predict_age_range(model_path=None, image_path=None, config=None):
    """
    Predicts the age range of a single OPG image.
    
    Parameters:
        model_path: Path to the saved model
        image_path: Path to the image
        config: Configuration object
        
    Returns:
        Prediction result
    """
    if config is None:
        config = Config()
        
    # Create predictor
    predictor = OPGPredictor(model_path=model_path, config=config)
    
    # Make prediction
    result = predictor.predict_age_range(image_path)
    
    # Print results
    print(f"\nPrediction for {os.path.basename(image_path)}:")
    print(f"Predicted Age Range: {result['predicted_age_range']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    return result

def batch_predict(model_path=None, directory=None, pattern="*.jpg", config=None):
    """
    Makes predictions on multiple OPG images.
    
    Parameters:
        model_path: Path to the saved model
        directory: Directory containing images
        pattern: Pattern to match image files
        config: Configuration object
        
    Returns:
        List of prediction results
    """
    if config is None:
        config = Config()
        
    # Create predictor
    predictor = OPGPredictor(model_path=model_path, config=config)
    
    # Find images
    if directory:
        image_paths = glob.glob(os.path.join(directory, pattern))
    else:
        print("No directory specified. Exiting.")
        return None
    
    if not image_paths:
        print(f"No images found matching {pattern} in {directory}")
        return None
    
    print(f"Found {len(image_paths)} images. Making predictions...")
    
    # Make predictions
    results = predictor.batch_predict(image_paths)
    
    # Print summary
    print("\nPrediction Results Summary:")
    for result in results:
        image_name = os.path.basename(result['image_path'])
        print(f"{image_name}: {result['predicted_age_range']} (Confidence: {result['confidence']:.4f})")
    
    return results

def find_similar_images(model_path=None, feature_extractor_path=None, query_image=None, dataset_dir=None, top_k=5, config=None):
    """
    Finds images similar to the query image.
    
    Parameters:
        model_path: Path to the saved Siamese model
        feature_extractor_path: Path to the feature extractor
        query_image: Path to the query image
        dataset_dir: Directory containing dataset images
        top_k: Number of similar images to return
        config: Configuration object
        
    Returns:
        List of similar images
    """
    if config is None:
        config = Config()
        
    # Create comparer
    comparer = SiameseImageComparer(
        model_path=model_path, 
        feature_extractor_path=feature_extractor_path,
        config=config
    )
    
    # Find similar images
    similar_images = comparer.find_similar_images(query_image, dataset_dir, top_k=top_k)
    
    # Print results
    print(f"\nImages similar to {os.path.basename(query_image)}:")
    for i, img in enumerate(similar_images):
        print(f"{i+1}. {os.path.basename(img['image_path'])} (Class: {img['class']}, Similarity: {img['similarity_score']:.4f})")
    
    return similar_images

def compare_two_images(model_path=None, feature_extractor_path=None, image1=None, image2=None, config=None):
    """
    Compares two OPG images and returns their similarity.
    
    Parameters:
        model_path: Path to the saved Siamese model
        feature_extractor_path: Path to the feature extractor
        image1: Path to the first image
        image2: Path to the second image
        config: Configuration object
        
    Returns:
        Similarity score
    """
    if config is None:
        config = Config()
        
    # Create comparer
    comparer = SiameseImageComparer(
        model_path=model_path, 
        feature_extractor_path=feature_extractor_path,
        config=config
    )
    
    # Compare images
    similarity = comparer.compare_images(image1, image2)
    
    # Print result
    print(f"\nSimilarity between {os.path.basename(image1)} and {os.path.basename(image2)}: {similarity:.4f}")
    
    return similarity

def main():
    """
    Main function to parse arguments and make predictions.
    """
    parser = argparse.ArgumentParser(description='OPG Age Classification Inference')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Inference mode')
    
    # Single prediction parser
    single_parser = subparsers.add_parser('predict', help='Predict age range for a single image')
    single_parser.add_argument('--model', type=str, default=None,
                             help='Path to the classification model')
    single_parser.add_argument('--image', type=str, required=True,
                             help='Path to the image')
    
    # Batch prediction parser
    batch_parser = subparsers.add_parser('batch', help='Predict age range for multiple images')
    batch_parser.add_argument('--model', type=str, default=None,
                            help='Path to the classification model')
    batch_parser.add_argument('--dir', type=str, required=True,
                            help='Directory containing images')
    batch_parser.add_argument('--pattern', type=str, default="*.jpg",
                            help='Pattern to match image files')
    
    # Similar images parser
    similar_parser = subparsers.add_parser('similar', help='Find similar images')
    similar_parser.add_argument('--model', type=str, default=None,
                              help='Path to the Siamese model')
    similar_parser.add_argument('--feature-extractor', type=str, default=None,
                              help='Path to the feature extractor')
    similar_parser.add_argument('--query', type=str, required=True,
                              help='Path to the query image')
    similar_parser.add_argument('--dir', type=str, required=True,
                              help='Directory containing dataset images')
    similar_parser.add_argument('--top-k', type=int, default=5,
                              help='Number of similar images to return')
    
    # Compare images parser
    compare_parser = subparsers.add_parser('compare', help='Compare two images')
    compare_parser.add_argument('--model', type=str, default=None,
                              help='Path to the Siamese model')
    compare_parser.add_argument('--feature-extractor', type=str, default=None,
                              help='Path to the feature extractor')
    compare_parser.add_argument('--image1', type=str, required=True,
                              help='Path to the first image')
    compare_parser.add_argument('--image2', type=str, required=True,
                              help='Path to the second image')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Run the appropriate function based on the mode
    if args.mode == 'predict':
        predict_age_range(model_path=args.model, image_path=args.image, config=config)
    elif args.mode == 'batch':
        batch_predict(model_path=args.model, directory=args.dir, pattern=args.pattern, config=config)
    elif args.mode == 'similar':
        find_similar_images(
            model_path=args.model, 
            feature_extractor_path=args.feature_extractor,
            query_image=args.query, 
            dataset_dir=args.dir, 
            top_k=args.top_k,
            config=config
        )
    elif args.mode == 'compare':
        compare_two_images(
            model_path=args.model, 
            feature_extractor_path=args.feature_extractor,
            image1=args.image1, 
            image2=args.image2,
            config=config
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 