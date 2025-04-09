"""
CNN model for sketch-to-code conversion.
This module contains the CNN model architecture for detecting UI components in sketches.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.models import Model
from typing import Tuple, List, Dict, Any, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
COMPONENT_CLASSES = [
    'button', 
    'input_field', 
    'checkbox',
    'navbar',
    'image_placeholder',
    'card',
    'dropdown',
    'toggle'
]

NUM_CLASSES = len(COMPONENT_CLASSES)
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join('model', 'saved_models', 'sketch2code_model.h5')


class SketchToCodeCNN:
    """CNN model for sketch-to-code conversion."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.is_model_loaded = False
        
    def build_model(self, input_shape: Tuple[int, int, int] = (*IMAGE_SIZE, 3), from_scratch: bool = True) -> None:
        """
        Build a CNN model architecture.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            from_scratch: Whether to build a custom CNN from scratch (True) or use transfer learning (False)
        """
        try:
            if from_scratch:
                # Build a custom CNN model from scratch
                logger.info("Building custom CNN model from scratch")
                inputs = keras.Input(shape=input_shape)
                
                # First convolutional block
                x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)
                x = layers.Dropout(0.1)(x)
                
                # Second convolutional block
                x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)
                x = layers.Dropout(0.2)(x)
                
                # Third convolutional block
                x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)
                x = layers.Dropout(0.3)(x)
                
                # Fourth convolutional block
                x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)
                x = layers.Dropout(0.4)(x)
                
                # Flatten and fully connected layers
                x = layers.Flatten()(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.5)(x)
                
                # For custom model, use higher learning rate
                learning_rate = 0.001
                
            else:
                # Use transfer learning with a pre-trained MobileNetV2 model
                logger.info("Building model with MobileNetV2 transfer learning")
                base_model = applications.MobileNetV2(
                    input_shape=input_shape,
                    include_top=False,
                    weights='imagenet'
                )
                
                # Freeze some of the base model layers to prevent overfitting
                for layer in base_model.layers[:100]:
                    layer.trainable = False
                
                # Build the model
                inputs = keras.Input(shape=input_shape)
                x = base_model(inputs, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dropout(0.2)(x)
                
                # For transfer learning, use lower learning rate
                learning_rate = 0.0001
            
            # Two outputs:
            # 1. Component classification
            # 2. Bounding box regression (x, y, width, height)
            classification_output = layers.Dense(NUM_CLASSES, activation='softmax', name='classification')(x)
            bounding_box_output = layers.Dense(4, activation='sigmoid', name='bounding_box')(x)
            
            self.model = keras.Model(inputs=inputs, outputs=[classification_output, bounding_box_output])
            
            # Compile the model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss={
                    'classification': 'categorical_crossentropy',
                    'bounding_box': 'mse'
                },
                metrics={
                    'classification': ['accuracy'],
                    'bounding_box': ['mae']
                }
            )
            
            logger.info(f"Model built successfully using {'custom CNN' if from_scratch else 'MobileNetV2 transfer learning'}")
            self.is_model_loaded = True
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            self.is_model_loaded = False
    
    def load_model(self, model_path: str = MODEL_PATH, from_scratch: bool = True) -> bool:
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model file
            from_scratch: Whether to build from scratch if loading fails
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
                self.is_model_loaded = True
                return True
            else:
                logger.warning(f"Model file not found at {model_path}. Building new model.")
                self.build_model(from_scratch=from_scratch)
                return self.is_model_loaded
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info(f"Building new model instead (from_scratch={from_scratch})")
            self.build_model(from_scratch=from_scratch)
            return self.is_model_loaded
    
    def predict(self, image_batch: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict UI components in an image.
        
        Args:
            image_batch: Batch of preprocessed images
            
        Returns:
            List of dictionaries containing component type and position
        """
        if not self.is_model_loaded or self.model is None:
            logger.error("Model not loaded. Please load model first.")
            return []
        
        try:
            # Make predictions
            classification_pred, bbox_pred = self.model.predict(image_batch)
            
            # Process predictions
            results = []
            
            for i in range(len(image_batch)):
                # Get class index with highest probability
                class_idx = np.argmax(classification_pred[i])
                confidence = float(classification_pred[i, class_idx])
                
                # Skip low confidence predictions
                if confidence < 0.3:
                    continue
                
                component_type = COMPONENT_CLASSES[class_idx]
                
                # Get bounding box
                x, y, width, height = bbox_pred[i]
                
                # Generate properties for component
                properties = self._generate_component_properties(component_type)
                
                # Add to results
                results.append({
                    'type': component_type,
                    'confidence': confidence,
                    'position': {
                        'x': float(x),
                        'y': float(y),
                        'width': float(width),
                        'height': float(height)
                    },
                    'properties': properties
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return []
    
    def save_model(self, model_path: str = MODEL_PATH) -> bool:
        """
        Save the model.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if model saved successfully, False otherwise
        """
        if not self.is_model_loaded or self.model is None:
            logger.error("No model to save.")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def _generate_component_properties(self, component_type: str) -> Dict[str, Any]:
        """
        Generate properties for a component. 
        
        This method creates reasonable defaults for components. In a real system,
        these would be predicted by the model or extracted from the image.
        
        Args:
            component_type: Type of UI component
            
        Returns:
            Dictionary of component properties
        """
        # We'll mimic the simulated model's properties generation for consistency
        if component_type == 'button':
            return {
                'text': self._get_random_button_text(),
                'variant': self._get_random_button_variant(),
                'size': self._get_random_from_array(['sm', 'md', 'lg'])
            }
        elif component_type == 'input_field':
            return {
                'placeholder': self._get_random_placeholder(),
                'type': self._get_random_from_array(['text', 'email', 'password', 'number']),
                'label': np.random.random() > 0.5
            }
        elif component_type == 'checkbox':
            return {
                'label': self._get_random_checkbox_label(),
                'checked': np.random.random() > 0.5
            }
        elif component_type == 'navbar':
            return {
                'brand': 'Brand Name',
                'links': int(2 + np.random.random() * 4),
                'dark': True
            }
        elif component_type == 'image_placeholder':
            return {
                'aspectRatio': self._get_random_from_array(['1x1', '4x3', '16x9']),
                'border': np.random.random() > 0.5
            }
        elif component_type == 'card':
            return {
                'title': 'Card Title',
                'text': True,
                'footer': np.random.random() > 0.5,
                'header': np.random.random() > 0.3
            }
        elif component_type == 'dropdown':
            return {
                'items': int(3 + np.random.random() * 5),
                'label': 'Dropdown'
            }
        elif component_type == 'toggle':
            return {
                'state': np.random.random() > 0.5,
                'label': 'Toggle Switch'
            }
        else:
            return {}
    
    def _get_random_button_text(self):
        texts = ['Submit', 'Save', 'Cancel', 'Send', 'Login', 'Register', 'Download', 'Upload', 'Next', 'Previous']
        return self._get_random_from_array(texts)
    
    def _get_random_button_variant(self):
        variants = ['primary', 'secondary', 'success', 'danger', 'warning', 'info', 'light', 'dark']
        return self._get_random_from_array(variants)
    
    def _get_random_placeholder(self):
        placeholders = ['Enter text...', 'Email address', 'Password', 'Search...', 'Your name', 'Phone number']
        return self._get_random_from_array(placeholders)
    
    def _get_random_checkbox_label(self):
        labels = ['Remember me', 'I agree to terms', 'Subscribe to newsletter', 'Save my information']
        return self._get_random_from_array(labels)
    
    def _get_random_from_array(self, array):
        return array[int(np.random.random() * len(array))]