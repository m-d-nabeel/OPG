#!/usr/bin/env python3
"""
Kaggle workflow script for Siamese OPG Age Classification.
This script handles the end-to-end workflow for Kaggle:
1. Data preparation (finding unique images, splitting into train/val)
2. Training the models
3. Evaluation and export of results
"""

import os
import argparse
import subprocess
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('kaggle_run.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, desc=None):
    """Run a command and log its output."""
    if desc:
        logger.info(f"Running: {desc}")
    
    logger.debug(f"Command: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        logger.info(f"✓ Completed successfully in {duration:.2f}s")
        logger.debug(result.stdout)
        return True
    else:
        logger.error(f"✗ Failed with code {result.returncode} after {duration:.2f}s")
        logger.error(f"Error: {result.stderr}")
        return False

def prepare_data(input_dir, working_dir, split_ratio=0.8, preprocess=True, update_config=True):
    """Prepare data for training."""
    cmd = [
        f"python -m siamese.data.prepare_data",
        f"--input {input_dir}",
        f"--output {working_dir}",
        f"--split-ratio {split_ratio}"
    ]
    
    if not preprocess:
        cmd.append("--no-preprocess")
        
    if update_config:
        cmd.append("--update-config")
        
    return run_command(" ".join(cmd), "Data preparation")

def train_models(mode='joint', extractor='hybrid', working_dir=None):
    """Train the models."""
    cmd = [
        f"python -m siamese.train.train",
        f"--mode {mode}",
        f"--extractor {extractor}"
    ]
    
    if working_dir:
        train_dir = os.path.join(working_dir, 'train')
        val_dir = os.path.join(working_dir, 'val')
        cmd.extend([
            f"--train_dir {train_dir}",
            f"--val_dir {val_dir}"
        ])
    
    return run_command(" ".join(cmd), f"Model training ({mode} mode)")

def export_results(working_dir, input_dir=None, mode=None, extractor=None, split_ratio=None):
    """Export results to Kaggle output directory."""
    output_dir = os.path.join(working_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy checkpoints
    checkpoint_dir = 'checkpoints'
    if os.path.exists(checkpoint_dir):
        run_command(f"cp -r {checkpoint_dir} {output_dir}/", "Exporting checkpoints")
    
    # Copy results
    results_dir = 'results'
    if os.path.exists(results_dir):
        run_command(f"cp -r {results_dir} {output_dir}/", "Exporting results")
    
    # Generate a summary report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"""
# Siamese OPG Age Classification - Run Summary

## Run Information
- Timestamp: {timestamp}
- Input directory: {input_dir or "Not specified"}
- Working directory: {working_dir}
- Training mode: {mode or "Not specified"}
- Feature extractor: {extractor or "Not specified"}
- Split ratio: {split_ratio or "Not specified"}

## Results
Check the 'results' folder for confusion matrices and training plots.
Trained models are available in the 'checkpoints' folder.
"""
    
    with open(os.path.join(output_dir, 'summary.md'), 'w') as f:
        f.write(summary)
    
    logger.info(f"Results exported to {output_dir}")
    return True

def main(args):
    """Main workflow function."""
    logger.info("=" * 80)
    logger.info(f"Starting Kaggle workflow - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Make sure the working directory exists
    os.makedirs(args.working, exist_ok=True)
    
    # 1. Prepare data
    if not args.skip_data_prep:
        success = prepare_data(
            args.input, 
            args.working,
            args.split_ratio,
            not args.no_preprocess,
            args.update_config
        )
        if not success:
            logger.error("Data preparation failed. Exiting.")
            return False
    
    # 2. Train models
    if not args.skip_training:
        success = train_models(args.mode, args.extractor, args.working)
        if not success:
            logger.error("Model training failed. Exiting.")
            return False
    
    # 3. Export results
    if not args.skip_export:
        success = export_results(
            working_dir=args.working,
            input_dir=args.input,
            mode=args.mode,
            extractor=args.extractor,
            split_ratio=args.split_ratio
        )
        if not success:
            logger.error("Exporting results failed.")
            return False
    
    logger.info("=" * 80)
    logger.info(f"Workflow completed successfully - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kaggle workflow for Siamese OPG Age Classification')
    parser.add_argument('--input', type=str, default='kaggle/input',
                        help='Input directory containing Kaggle dataset')
    parser.add_argument('--working', type=str, default='kaggle/working',
                        help='Working directory for processed data')
    parser.add_argument('--mode', type=str, default='joint', choices=['separate', 'joint'],
                        help='Training mode: "separate" or "joint"')
    parser.add_argument('--extractor', type=str, default='hybrid', choices=['cnn', 'vit', 'hybrid'],
                        help='Type of feature extractor to use')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                        help='Ratio of training to total data')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Skip image preprocessing')
    parser.add_argument('--update-config', action='store_true', default=True,
                        help='Update the config file with the new data paths')
    parser.add_argument('--skip-data-prep', action='store_true',
                        help='Skip data preparation step')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--skip-export', action='store_true',
                        help='Skip exporting results')
    
    args = parser.parse_args()
    
    success = main(args)
    exit(0 if success else 1) 