"""
Preprocessing module for sketch images.
This module contains functions to preprocess images for the CNN model.
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (224, 224)  # Standard input size for many CNN models


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image as numpy array or None if loading fails
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return None
            
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None
            
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load an image from binary data.
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        Loaded image as numpy array or None if loading fails
    """
    try:
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image from bytes")
            return None
            
        # Convert BGR to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        logger.error(f"Error loading image from bytes: {str(e)}")
        return None


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess an image for the CNN model.
    
    Args:
        image: Input image as numpy array
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image
    """
    # Resize image
    resized = cv2.resize(image, target_size)
    
    # Convert to grayscale (optional depending on your model)
    # We'll keep the color information for now, but comment this option
    # gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    # gray = np.expand_dims(gray, axis=-1)  # Add channel dimension
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


def extract_image_regions(image: np.ndarray, regions: List[Dict[str, Any]]) -> List[np.ndarray]:
    """
    Extract regions from an image based on bounding box coordinates.
    
    Args:
        image: Input image
        regions: List of region dictionaries with 'x', 'y', 'width', 'height' keys
        
    Returns:
        List of image regions
    """
    extracted_regions = []
    
    height, width = image.shape[:2]
    
    for region in regions:
        # Get coordinates as absolute pixels
        x = int(region['x'] * width)
        y = int(region['y'] * height)
        w = int(region['width'] * width)
        h = int(region['height'] * height)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Extract region
        region_img = image[y:y+h, x:x+w]
        
        # Add to list
        extracted_regions.append(region_img)
    
    return extracted_regions


def batch_preprocess(images: List[np.ndarray], target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Preprocess a batch of images.
    
    Args:
        images: List of input images
        target_size: Target size for resizing
        
    Returns:
        Batch of preprocessed images as numpy array
    """
    processed_images = []
    
    for image in images:
        processed = preprocess_image(image, target_size)
        processed_images.append(processed)
    
    # Stack into batch
    batch = np.stack(processed_images, axis=0)
    
    return batch


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """
    Apply data augmentation to an image.
    
    Args:
        image: Input image
        
    Returns:
        List of augmented images
    """
    augmented = []
    
    # Original image
    augmented.append(image)
    
    # Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented.append(flipped)
    
    # Rotation
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    rotated = cv2.warpAffine(image, rot_matrix, (cols, rows))
    augmented.append(rotated)
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
    augmented.append(bright)
    
    return augmented