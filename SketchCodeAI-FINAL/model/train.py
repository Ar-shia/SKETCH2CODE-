"""
Training script for the sketch-to-code CNN model.
This script trains the CNN model using synthetic data.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import json
from typing import Tuple, List, Dict, Any

from .cnn_model import SketchToCodeCNN, COMPONENT_CLASSES, IMAGE_SIZE
from .preprocessing import preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 16  # Smaller batch size for better convergence when training from scratch
EPOCHS = 100  # More epochs for training from scratch
DATA_DIR = os.path.join('model', 'training_data')


def generate_synthetic_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for model training.
    
    For demonstration purposes, this function creates synthetic data that mimics
    UI component sketches. In a real-world scenario, you would use a dataset of
    actual UI wireframes labeled with component types and positions.
    
    Args:
        num_samples: Number of synthetic samples to generate
        
    Returns:
        Tuple of (images, class_labels, bounding_boxes)
    """
    logger.info(f"Generating {num_samples} synthetic training samples...")
    
    # Create empty arrays for data
    images = np.zeros((num_samples, *IMAGE_SIZE, 3), dtype=np.float32)
    class_labels = np.zeros((num_samples, len(COMPONENT_CLASSES)), dtype=np.float32)
    bounding_boxes = np.zeros((num_samples, 4), dtype=np.float32)  # (x, y, width, height)
    
    # Create directory for sample images
    os.makedirs(os.path.join(DATA_DIR, 'samples'), exist_ok=True)
    
    for i in range(num_samples):
        # Progress log
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        # Create a blank canvas
        image = np.ones((*IMAGE_SIZE, 3), dtype=np.uint8) * 255
        
        # Randomly select a component class
        class_idx = np.random.randint(0, len(COMPONENT_CLASSES))
        component_type = COMPONENT_CLASSES[class_idx]
        
        # One-hot encode the class
        class_labels[i, class_idx] = 1.0
        
        # Generate random position and size
        x = np.random.uniform(0.1, 0.7)
        y = np.random.uniform(0.1, 0.7)
        width = np.random.uniform(0.1, 0.3)
        height = np.random.uniform(0.1, 0.3)
        
        # Ensure it fits within image bounds
        width = min(width, 1.0 - x)
        height = min(height, 1.0 - y)
        
        # Store bounding box
        bounding_boxes[i] = [x, y, width, height]
        
        # Convert to pixel coordinates
        x_px = int(x * IMAGE_SIZE[1])
        y_px = int(y * IMAGE_SIZE[0])
        width_px = int(width * IMAGE_SIZE[1])
        height_px = int(height * IMAGE_SIZE[0])
        
        # Draw the component
        if component_type == 'button':
            # Draw a rectangle with text for button
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (200, 200, 200), -1)
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (100, 100, 100), 2)
            
            # Add text
            text = "Button"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = x_px + (width_px - text_size[0]) // 2
            text_y = y_px + (height_px + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), font, 0.5, (50, 50, 50), 1)
            
        elif component_type == 'input_field':
            # Draw a rectangle with a line for input field
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (240, 240, 240), -1)
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (100, 100, 100), 1)
            
            # Add placeholder text
            text = "Input..."
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = x_px + 5
            text_y = y_px + (height_px + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), font, 0.5, (150, 150, 150), 1)
            
        elif component_type == 'checkbox':
            # Draw a square with a check mark
            box_size = min(width_px, height_px)
            cv2.rectangle(image, (x_px, y_px), (x_px + box_size, y_px + box_size), (240, 240, 240), -1)
            cv2.rectangle(image, (x_px, y_px), (x_px + box_size, y_px + box_size), (100, 100, 100), 1)
            
            # Add label text
            text = "Checkbox"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_x = x_px + box_size + 5
            text_y = y_px + box_size // 2
            cv2.putText(image, text, (text_x, text_y), font, 0.5, (50, 50, 50), 1)
            
        elif component_type == 'navbar':
            # Draw a rectangle at the top for navbar
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (200, 200, 200), -1)
            
            # Add brand text
            brand = "Brand"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, brand, (x_px + 10, y_px + height_px // 2), font, 0.6, (50, 50, 50), 1)
            
            # Add menu items
            menu_items = ["Home", "About", "Contact"]
            menu_x = x_px + width_px // 2
            for item in menu_items:
                cv2.putText(image, item, (menu_x, y_px + height_px // 2), font, 0.5, (80, 80, 80), 1)
                menu_x += 70
                
        elif component_type == 'image_placeholder':
            # Draw a rectangle with an 'X' for image placeholder
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (240, 240, 240), -1)
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (100, 100, 100), 1)
            
            # Draw an X across the rectangle
            cv2.line(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (150, 150, 150), 1)
            cv2.line(image, (x_px, y_px + height_px), (x_px + width_px, y_px), (150, 150, 150), 1)
            
            # Add text
            text = "Image"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = x_px + (width_px - text_size[0]) // 2
            text_y = y_px + (height_px + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), font, 0.5, (100, 100, 100), 1)
            
        elif component_type == 'card':
            # Draw a card with header, content area, and footer
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (250, 250, 250), -1)
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px), (100, 100, 100), 1)
            
            # Header
            header_height = height_px // 5
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + header_height), (230, 230, 230), -1)
            cv2.line(image, (x_px, y_px + header_height), (x_px + width_px, y_px + header_height), (150, 150, 150), 1)
            
            # Title
            text = "Card"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (x_px + 10, y_px + header_height + 20), font, 0.5, (50, 50, 50), 1)
            
            # Content lines
            line_y = y_px + header_height + 40
            for _ in range(2):
                cv2.line(image, (x_px + 10, line_y), (x_px + width_px - 10, line_y), (200, 200, 200), 1)
                line_y += 15
            
        elif component_type == 'dropdown':
            # Draw dropdown button and menu
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px // 3), (200, 200, 200), -1)
            cv2.rectangle(image, (x_px, y_px), (x_px + width_px, y_px + height_px // 3), (100, 100, 100), 1)
            
            # Add dropdown text and arrow
            text = "Dropdown"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (x_px + 10, y_px + height_px // 6 + 5), font, 0.5, (50, 50, 50), 1)
            cv2.putText(image, "▼", (x_px + width_px - 20, y_px + height_px // 6 + 5), font, 0.5, (50, 50, 50), 1)
            
            # Draw dropdown menu
            menu_y = y_px + height_px // 3
            cv2.rectangle(image, (x_px, menu_y), (x_px + width_px, y_px + height_px), (240, 240, 240), -1)
            cv2.rectangle(image, (x_px, menu_y), (x_px + width_px, y_px + height_px), (100, 100, 100), 1)
            
            # Add menu items
            item_y = menu_y + 15
            for item in ["Item 1", "Item 2", "Item 3"]:
                cv2.putText(image, item, (x_px + 10, item_y), font, 0.4, (80, 80, 80), 1)
                item_y += 20
                
        elif component_type == 'toggle':
            # Draw toggle switch
            toggle_width = width_px
            toggle_height = height_px // 3
            toggle_y = y_px + (height_px - toggle_height) // 2
            
            # Background track
            cv2.rectangle(image, (x_px, toggle_y), (x_px + toggle_width, toggle_y + toggle_height), (200, 200, 200), -1)
            cv2.rectangle(image, (x_px, toggle_y), (x_px + toggle_width, toggle_y + toggle_height), (100, 100, 100), 1)
            
            # Toggle indicator
            indicator_size = toggle_height
            indicator_x = x_px + toggle_width - indicator_size - 2
            cv2.rectangle(image, (indicator_x, toggle_y), (indicator_x + indicator_size, toggle_y + toggle_height), (250, 250, 250), -1)
            cv2.rectangle(image, (indicator_x, toggle_y), (indicator_x + indicator_size, toggle_y + toggle_height), (100, 100, 100), 1)
            
            # Add label
            text = "Toggle"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (x_px, y_px - 5), font, 0.5, (50, 50, 50), 1)
        
        # Apply a sketch effect (optional)
        if np.random.random() > 0.3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply a slight blur
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold to create sketch-like effect
            sketch = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Convert back to RGB
            sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
            
            # Blend with original
            alpha = 0.7
            image = cv2.addWeighted(sketch_rgb, alpha, image, 1-alpha, 0)
        
        # Add some noise
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        # Save a sample of images for verification
        if i < 10:
            sample_path = os.path.join(DATA_DIR, 'samples', f"sample_{i}_{component_type}.png")
            cv2.imwrite(sample_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Preprocess and add to dataset
        preprocessed = preprocess_image(image)
        images[i] = preprocessed
    
    logger.info("Synthetic data generation complete")
    return images, class_labels, bounding_boxes


def save_metadata(class_labels: np.ndarray, bounding_boxes: np.ndarray, save_dir: str) -> None:
    """
    Save metadata for the synthetic dataset.
    
    Args:
        class_labels: One-hot encoded class labels
        bounding_boxes: Bounding box coordinates
        save_dir: Directory to save metadata
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metadata = []
    for i in range(len(class_labels)):
        class_idx = np.argmax(class_labels[i])
        component_type = COMPONENT_CLASSES[class_idx]
        bbox = bounding_boxes[i].tolist()
        
        metadata.append({
            "id": i,
            "component_type": component_type,
            "bbox": bbox
        })
    
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def train_model(
    model_instance: SketchToCodeCNN, 
    images: np.ndarray, 
    class_labels: np.ndarray, 
    bounding_boxes: np.ndarray,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS
) -> Any:
    """
    Train the CNN model.
    
    Args:
        model_instance: Instance of SketchToCodeCNN
        images: Training images
        class_labels: One-hot encoded class labels
        bounding_boxes: Bounding box coordinates
        batch_size: Batch size for training
        epochs: Number of epochs to train
        
    Returns:
        Training history
    """
    logger.info("Splitting data into training and validation sets...")
    
    # Split into training and validation sets
    X_train, X_val, y_class_train, y_class_val, y_bbox_train, y_bbox_val = train_test_split(
        images, class_labels, bounding_boxes, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    
    # Data augmentation (optional)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # Create model if not built yet
    if model_instance.model is None:
        model_instance.build_model()
    
    # Prepare callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join('model', 'checkpoints', 'sketch2code_model_{epoch:02d}.h5'),
            save_best_only=True,
            monitor='val_classification_accuracy',
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Ensure checkpoint directory exists
    os.makedirs(os.path.join('model', 'checkpoints'), exist_ok=True)
    
    logger.info("Starting model training...")
    
    # Train the model
    history = model_instance.model.fit(
        X_train,
        {
            'classification': y_class_train,
            'bounding_box': y_bbox_train
        },
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(
            X_val, 
            {
                'classification': y_class_val,
                'bounding_box': y_bbox_val
            }
        ),
        callbacks=callbacks
    )
    
    logger.info("Training complete")
    
    # Save the final model
    model_instance.save_model()
    
    return history


def plot_training_history(history: Any) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: Training history
    """
    # Create directory for plots
    plots_dir = os.path.join('model', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Classification accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['classification_accuracy'])
    plt.plot(history.history['val_classification_accuracy'])
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Bounding box mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(history.history['bounding_box_mae'])
    plt.plot(history.history['val_bounding_box_mae'])
    plt.title('Bounding Box MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png'))
    
    # Loss plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['classification_loss'])
    plt.plot(history.history['val_classification_loss'])
    plt.title('Classification Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['bounding_box_loss'])
    plt.plot(history.history['val_bounding_box_loss'])
    plt.title('Bounding Box Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_loss.png'))
    
    logger.info(f"Training plots saved to {plots_dir}")


def main(from_scratch=False, num_samples=1000, epochs=None, batch_size=None):
    """
    Main function to train the model.
    
    Args:
        from_scratch: Whether to train a custom CNN from scratch (True) or use transfer learning (False)
        num_samples: Number of synthetic samples to generate
        epochs: Number of training epochs (None uses default value)
        batch_size: Batch size for training (None uses default value)
    """
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info(f"Training mode: {'From scratch' if from_scratch else 'Transfer learning'}")
    
    # Generate synthetic data
    images, class_labels, bounding_boxes = generate_synthetic_data(num_samples=num_samples)
    
    # Save metadata
    save_metadata(class_labels, bounding_boxes, DATA_DIR)
    
    # Create model instance and select architecture
    model = SketchToCodeCNN()
    
    # Configure model to build from scratch or use transfer learning
    if from_scratch:
        logger.info("Building custom CNN architecture from scratch")
        model.build_model(from_scratch=True)
    else:
        logger.info("Using transfer learning with MobileNetV2")
        model.build_model(from_scratch=False)
    
    # Set training parameters
    training_epochs = epochs if epochs is not None else EPOCHS
    training_batch_size = batch_size if batch_size is not None else BATCH_SIZE
    
    logger.info(f"Training with {training_epochs} epochs and batch size {training_batch_size}")
    
    # Train the model
    history = train_model(
        model, 
        images, 
        class_labels, 
        bounding_boxes,
        batch_size=training_batch_size,
        epochs=training_epochs
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the final model
    model.save_model()
    
    logger.info("Model training and saving complete")


if __name__ == "__main__":
    main()