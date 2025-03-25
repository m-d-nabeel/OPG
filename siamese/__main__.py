import argparse

def main():
    """
    Main entry point for the siamese package.
    """
    parser = argparse.ArgumentParser(
        description='Siamese OPG Age Classification',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train models')
    
    # Inference parser
    inference_parser = subparsers.add_parser('predict', help='Make predictions')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Import here to avoid loading modules unnecessarily
        from siamese.train.train import main as train_main
        train_main()
    elif args.command == 'predict':
        # Import here to avoid loading modules unnecessarily
        from siamese.inference.predict import main as predict_main
        predict_main()
    else:
        # Print help
        parser.print_help()

if __name__ == '__main__':
    main() 