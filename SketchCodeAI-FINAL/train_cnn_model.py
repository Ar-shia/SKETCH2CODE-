"""
Standalone script to train the sketch-to-code CNN model.
"""

import os
import logging
import argparse
from model.train import main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train the Sketch-to-Code CNN model')
    parser.add_argument('--from-scratch', action='store_true', help='Train a model from scratch instead of using transfer learning')
    parser.add_argument('--samples', type=int, default=1000, help='Number of synthetic training samples to generate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Run the training process with command line arguments
    main(from_scratch=args.from_scratch, 
         num_samples=args.samples, 
         epochs=args.epochs, 
         batch_size=args.batch_size)