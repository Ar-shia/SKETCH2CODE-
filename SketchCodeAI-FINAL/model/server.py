"""
Server interface for the sketch-to-code CNN model.
This module contains functions to interface with the Flask server.
"""

import os
import base64
import tempfile
from typing import Tuple, Dict, Any, Optional
import logging
import json

from .detector import UIComponentDetector

# Configure logging
logger = logging.getLogger(__name__)

# Global detector instance
detector = UIComponentDetector()


def initialize_model(from_scratch: bool = True) -> bool:
    """
    Initialize the model for use in the server.
    
    Args:
        from_scratch: Whether to use a custom CNN (True) or transfer learning (False)
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        success = detector.initialize(from_scratch=from_scratch)
        
        # Log which model type is being used
        model_type = "custom CNN (trained from scratch)" if from_scratch else "MobileNetV2 (transfer learning)"
        logger.info(f"Initialized model using {model_type}")
        
        return success
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return False


def process_image(image_data: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Process an image and detect UI components.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        Tuple of (success, result)
    """
    try:
        # Detect components
        components = detector.detect_from_base64(image_data)
        
        if not components:
            return False, {"error": "No UI components detected in the image"}
        
        # Generate HTML and CSS code
        html_code = detector.generate_html(components)
        css_code = detector.generate_css(components)
        
        # Return results
        return True, {
            "success": True,
            "components": components,
            "html_code": html_code,
            "css_code": css_code
        }
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return False, {"error": f"Error processing image: {str(e)}"}


def process_image_file(file_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Process an image file and detect UI components.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (success, result)
    """
    try:
        # Detect components
        components = detector.detect_from_file(file_path)
        
        if not components:
            return False, {"error": "No UI components detected in the image"}
        
        # Generate HTML and CSS code
        html_code = detector.generate_html(components)
        css_code = detector.generate_css(components)
        
        # Return results
        return True, {
            "success": True,
            "components": components,
            "html_code": html_code,
            "css_code": css_code
        }
    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}")
        return False, {"error": f"Error processing image file: {str(e)}"}


def save_temp_image(image_data: str) -> Optional[str]:
    """
    Save base64 encoded image to a temporary file.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        Path to the temporary file or None if error
    """
    try:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.png')
        
        # Decode base64 data
        if ',' in image_data:
            # Handle data URIs (e.g., "data:image/jpeg;base64,...")
            _, image_data = image_data.split(',', 1)
        
        image_bytes = base64.b64decode(image_data)
        
        # Write to the file
        with os.fdopen(fd, 'wb') as temp:
            temp.write(image_bytes)
        
        return temp_path
    except Exception as e:
        logger.error(f"Error saving temporary image: {str(e)}")
        return None