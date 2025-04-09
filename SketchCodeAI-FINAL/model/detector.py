"""
UI Component detector module.
This module contains functions to detect UI components in sketches and generate HTML/CSS code.
"""

import os
import base64
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import logging
import json
from PIL import Image
import io

from .preprocessing import preprocess_image, load_image, load_image_from_bytes
from .cnn_model import SketchToCodeCNN, COMPONENT_CLASSES

# Configure logging
logger = logging.getLogger(__name__)


class UIComponentDetector:
    """Detector for UI components in sketches."""
    
    def __init__(self):
        """Initialize the detector."""
        self.model = SketchToCodeCNN()
        self.is_initialized = False
        self.unique_component_ids = {}
    
    def initialize(self, from_scratch: bool = True) -> bool:
        """
        Initialize the detector and load the model.
        
        Args:
            from_scratch: Whether to use a custom CNN (True) or transfer learning (False)
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create model directory structure if it doesn't exist
            os.makedirs(os.path.join('model', 'saved_models'), exist_ok=True)
            
            # Load the model with the specified architecture type
            self.is_initialized = self.model.load_model(from_scratch=from_scratch)
            
            model_type = "custom CNN (trained from scratch)" if from_scratch else "MobileNetV2 (transfer learning)"
            if self.is_initialized:
                logger.info(f"Model initialized successfully using {model_type}")
            else:
                logger.warning(f"Failed to initialize model with {model_type}")
                
            return self.is_initialized
        except Exception as e:
            logger.error(f"Error initializing detector: {str(e)}")
            return False
    
    def detect_from_file(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect UI components in an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected components
        """
        if not self.is_initialized:
            logger.error("Detector not initialized. Please call initialize() first.")
            return []
        
        try:
            # Load and preprocess the image
            image = load_image(image_path)
            if image is None:
                return []
                
            preprocessed = preprocess_image(image)
            
            # Add batch dimension
            input_batch = np.expand_dims(preprocessed, axis=0)
            
            # Make prediction
            components = self.model.predict(input_batch)
            
            return components
        except Exception as e:
            logger.error(f"Error detecting components: {str(e)}")
            return []
    
    def detect_from_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Detect UI components in an image from binary data.
        
        Args:
            image_bytes: Binary image data
            
        Returns:
            List of detected components
        """
        if not self.is_initialized:
            logger.error("Detector not initialized. Please call initialize() first.")
            return []
        
        try:
            # Load and preprocess the image
            image = load_image_from_bytes(image_bytes)
            if image is None:
                return []
                
            preprocessed = preprocess_image(image)
            
            # Add batch dimension
            input_batch = np.expand_dims(preprocessed, axis=0)
            
            # Make prediction
            components = self.model.predict(input_batch)
            
            return components
        except Exception as e:
            logger.error(f"Error detecting components from bytes: {str(e)}")
            return []
    
    def detect_from_base64(self, base64_data: str) -> List[Dict[str, Any]]:
        """
        Detect UI components in a base64 encoded image.
        
        Args:
            base64_data: Base64 encoded image data
            
        Returns:
            List of detected components
        """
        try:
            # Decode base64 data
            if ',' in base64_data:
                # Handle data URIs (e.g., "data:image/jpeg;base64,...")
                _, base64_data = base64_data.split(',', 1)
            
            image_bytes = base64.b64decode(base64_data)
            return self.detect_from_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Error detecting components from base64: {str(e)}")
            return []
    
    def generate_html(self, components: List[Dict[str, Any]]) -> str:
        """
        Generate HTML code based on detected components.
        
        Args:
            components: List of detected UI components
            
        Returns:
            Generated HTML code
        """
        # Start with a basic Bootstrap template
        html_code = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Wireframe</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="styles.css" rel="stylesheet">
</head>
<body>
    <div class="container">
"""
        
        # Sort components by Y position to maintain layout order
        sorted_components = sorted(components, key=lambda c: c['position']['y'])
        
        # Extract navbar if present
        navbar = next((c for c in sorted_components if c['type'] == 'navbar'), None)
        other_components = [c for c in sorted_components if c['type'] != 'navbar']
        
        # Add navbar first if present
        if navbar:
            html_code += self._generate_navbar_code(navbar)
        
        # Add container for the rest of the components
        html_code += """
        <div class="row mt-4">
"""
        
        # Add other components based on their position
        for component in other_components:
            col_size = self._determine_column_size(component)
            html_code += f"""            <div class="{col_size} mb-3">\n"""
            html_code += self._generate_component_code(component)
            html_code += """            </div>\n"""
        
        # Close containers
        html_code += """        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""
        
        return html_code
    
    def generate_css(self, components: List[Dict[str, Any]]) -> str:
        """
        Generate CSS code based on detected components.
        
        Args:
            components: List of detected UI components
            
        Returns:
            Generated CSS code
        """
        # Basic CSS with custom styling for components
        css_code = """/* 
 * Generated CSS for wireframe
 * Created by Sketch to Code converter
 */

/* General styles */
body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    padding-bottom: 20px;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

/* Custom component styles */
"""
        
        # Add specific styles for different component types
        component_counts = {}
        
        for component in components:
            component_type = component['type']
            
            # Create unique class names for each component type
            if component_type not in component_counts:
                component_counts[component_type] = 1
            else:
                component_counts[component_type] += 1
            
            # Store unique ID for this component
            if component_type not in self.unique_component_ids:
                self.unique_component_ids[component_type] = 1
            
            component_id = f"{component_type}-{self.unique_component_ids[component_type]}"
            self.unique_component_ids[component_type] += 1
            
            # Add component-specific CSS
            if component_type == 'button':
                css_code += self._generate_button_css(component, component_id)
            elif component_type == 'input_field':
                css_code += self._generate_input_css(component, component_id)
            elif component_type == 'navbar':
                css_code += self._generate_navbar_css(component, component_id)
            elif component_type == 'image_placeholder':
                css_code += self._generate_image_placeholder_css(component, component_id)
            elif component_type == 'card':
                css_code += self._generate_card_css(component, component_id)
            elif component_type == 'dropdown':
                css_code += self._generate_dropdown_css(component, component_id)
            elif component_type == 'toggle':
                css_code += self._generate_toggle_css(component, component_id)
        
        # Add responsive styling
        css_code += """
/* Responsive styles */
@media (max-width: 768px) {
    .row > div[class^="col"] {
        margin-bottom: 1rem;
    }
    
    .navbar-brand {
        font-size: 1.2rem;
    }
}
"""
        
        return css_code
    
    def _determine_column_size(self, component: Dict[str, Any]) -> str:
        """
        Determine appropriate Bootstrap column size based on component position.
        
        Args:
            component: Component data
            
        Returns:
            Bootstrap column class
        """
        width = component['position']['width']
        
        if width < 0.3:
            return "col-md-4"
        elif width < 0.6:
            return "col-md-6"
        else:
            return "col-md-12"
    
    def _generate_component_code(self, component: Dict[str, Any]) -> str:
        """
        Generate code for a specific component.
        
        Args:
            component: Component data
            
        Returns:
            HTML code for the component
        """
        component_type = component['type']
        
        if component_type == 'button':
            return self._generate_button_code(component)
        elif component_type == 'input_field':
            return self._generate_input_code(component)
        elif component_type == 'checkbox':
            return self._generate_checkbox_code(component)
        elif component_type == 'navbar':
            return self._generate_navbar_code(component)
        elif component_type == 'image_placeholder':
            return self._generate_image_code(component)
        elif component_type == 'card':
            return self._generate_card_code(component)
        elif component_type == 'dropdown':
            return self._generate_dropdown_code(component)
        elif component_type == 'toggle':
            return self._generate_toggle_code(component)
        else:
            return """                <!-- Unknown component type -->\n"""
    
    def _generate_button_code(self, component: Dict[str, Any]) -> str:
        """Generate button HTML code."""
        props = component['properties']
        variant = props.get('variant', 'primary')
        size = props.get('size', '')
        size_class = f"btn-{size}" if size else ''
        text = props.get('text', 'Button')
        
        return f"""                <button type="button" class="btn btn-{variant} {size_class}">{text}</button>\n"""
    
    def _generate_input_code(self, component: Dict[str, Any]) -> str:
        """Generate input field HTML code."""
        props = component['properties']
        input_type = props.get('type', 'text')
        placeholder = props.get('placeholder', '')
        has_label = props.get('label', False)
        
        if has_label:
            label_text = placeholder or f"Enter {input_type}"
            input_id = f"input_{int(np.random.random() * 1000)}"
            
            return f"""                <div class="mb-3">
                    <label for="{input_id}" class="form-label">{label_text}</label>
                    <input type="{input_type}" class="form-control" id="{input_id}" placeholder="{placeholder}">
                </div>\n"""
        else:
            return f"""                <input type="{input_type}" class="form-control" placeholder="{placeholder}">\n"""
    
    def _generate_checkbox_code(self, component: Dict[str, Any]) -> str:
        """Generate checkbox HTML code."""
        props = component['properties']
        label = props.get('label', 'Checkbox')
        checked = 'checked' if props.get('checked', False) else ''
        checkbox_id = f"check_{int(np.random.random() * 1000)}"
        
        return f"""                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="{checkbox_id}" {checked}>
                    <label class="form-check-label" for="{checkbox_id}">
                        {label}
                    </label>
                </div>\n"""
    
    def _generate_navbar_code(self, component: Dict[str, Any]) -> str:
        """Generate navbar HTML code."""
        props = component['properties']
        brand = props.get('brand', 'Brand')
        num_links = props.get('links', 3)
        dark = props.get('dark', True)
        theme = 'navbar-dark bg-dark' if dark else 'navbar-light bg-light'
        
        links = ''
        link_names = ['Home', 'Features', 'Pricing', 'About', 'Contact', 'Services', 'Blog']
        
        for i in range(min(num_links, len(link_names))):
            active = 'active' if i == 0 else ''
            links += f"""                    <li class="nav-item">
                        <a class="nav-link {active}" href="#">{link_names[i]}</a>
                    </li>\n"""
        
        return f"""    <nav class="navbar navbar-expand-lg {theme}">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">{brand}</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
{links}                </ul>
            </div>
        </div>
    </nav>\n"""
    
    def _generate_image_code(self, component: Dict[str, Any]) -> str:
        """Generate image placeholder HTML code."""
        props = component['properties']
        aspect_ratio = props.get('aspectRatio', '4x3')
        border = 'border rounded' if props.get('border', False) else ''
        
        return f"""                <div class="ratio ratio-{aspect_ratio} {border} bg-light d-flex align-items-center justify-content-center text-muted">
                    <div>
                        <i class="fas fa-image fa-2x mb-2"></i>
                        <p>Image Placeholder</p>
                    </div>
                </div>\n"""
    
    def _generate_card_code(self, component: Dict[str, Any]) -> str:
        """Generate card HTML code."""
        props = component['properties']
        title = props.get('title', 'Card Title')
        has_text = props.get('text', True)
        has_footer = props.get('footer', False)
        has_header = props.get('header', False)
        
        card_html = """                <div class="card">\n"""
        
        if has_header:
            card_html += """                    <div class="card-header">
                        Featured
                    </div>\n"""
        
        card_html += """                    <div class="card-body">\n"""
        card_html += f"""                        <h5 class="card-title">{title}</h5>\n"""
        
        if has_text:
            card_html += """                        <p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
                        <a href="#" class="btn btn-primary">Go somewhere</a>\n"""
        
        card_html += """                    </div>\n"""
        
        if has_footer:
            card_html += """                    <div class="card-footer text-muted">
                        2 days ago
                    </div>\n"""
        
        card_html += """                </div>\n"""
        
        return card_html
    
    def _generate_dropdown_code(self, component: Dict[str, Any]) -> str:
        """Generate dropdown HTML code."""
        props = component['properties']
        label = props.get('label', 'Dropdown')
        num_items = props.get('items', 3)
        
        dropdown_items = ''
        for i in range(num_items):
            dropdown_items += f"""                        <li><a class="dropdown-item" href="#">Action {i+1}</a></li>\n"""
        
        return f"""                <div class="dropdown">
                    <button class="btn btn-secondary dropdown-toggle" type="button" id="dropdownMenuButton" data-bs-toggle="dropdown" aria-expanded="false">
                        {label}
                    </button>
                    <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton">
{dropdown_items}                    </ul>
                </div>\n"""
    
    def _generate_toggle_code(self, component: Dict[str, Any]) -> str:
        """Generate toggle switch HTML code."""
        props = component['properties']
        label = props.get('label', 'Toggle Switch')
        state = 'checked' if props.get('state', False) else ''
        toggle_id = f"toggle_{int(np.random.random() * 1000)}"
        
        return f"""                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="{toggle_id}" {state}>
                    <label class="form-check-label" for="{toggle_id}">{label}</label>
                </div>\n"""
    
    def _generate_button_css(self, component: Dict[str, Any], component_id: str) -> str:
        """Generate button CSS code."""
        props = component['properties']
        variant = props.get('variant', 'primary')
        
        return f"""
/* Button styles - {component_id} */
.btn-{variant} {{
    transition: all 0.3s ease;
}}

.btn-{variant}:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}}
"""
    
    def _generate_input_css(self, component: Dict[str, Any], component_id: str) -> str:
        """Generate input field CSS code."""
        return f"""
/* Input field styles - {component_id} */
.form-control:focus {{
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
    border-color: #80bdff;
}}

.form-label {{
    font-weight: 500;
    margin-bottom: 0.5rem;
}}
"""
    
    def _generate_navbar_css(self, component: Dict[str, Any], component_id: str) -> str:
        """Generate navbar CSS code."""
        props = component['properties']
        dark = props.get('dark', True)
        
        return f"""
/* Navbar styles - {component_id} */
.navbar {{
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 1.5rem;
}}

.navbar-brand {{
    font-weight: bold;
    letter-spacing: 0.5px;
}}

.nav-link {{
    position: relative;
}}

.nav-link.active:after {{
    content: '';
    position: absolute;
    left: 0;
    bottom: -2px;
    width: 100%;
    height: 2px;
    background-color: {('#fff' if dark else '#007bff')};
}}
"""
    
    def _generate_image_placeholder_css(self, component: Dict[str, Any], component_id: str) -> str:
        """Generate image placeholder CSS code."""
        return f"""
/* Image placeholder styles - {component_id} */
.ratio {{
    transition: all 0.3s ease;
    overflow: hidden;
}}

.ratio:hover {{
    opacity: 0.9;
    cursor: pointer;
}}

.ratio i, .ratio p {{
    opacity: 0.7;
}}
"""
    
    def _generate_card_css(self, component: Dict[str, Any], component_id: str) -> str:
        """Generate card CSS code."""
        return f"""
/* Card styles - {component_id} */
.card {{
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 1.5rem;
}}

.card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
}}

.card-title {{
    font-weight: 600;
    margin-bottom: 0.75rem;
}}
"""
    
    def _generate_dropdown_css(self, component: Dict[str, Any], component_id: str) -> str:
        """Generate dropdown CSS code."""
        return f"""
/* Dropdown styles - {component_id} */
.dropdown-menu {{
    border-radius: 0.25rem;
    box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
    border: none;
    padding: 0.5rem;
}}

.dropdown-item {{
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
}}

.dropdown-item:hover {{
    background-color: #f8f9fa;
}}
"""
    
    def _generate_toggle_css(self, component: Dict[str, Any], component_id: str) -> str:
        """Generate toggle switch CSS code."""
        return f"""
/* Toggle switch styles - {component_id} */
.form-switch .form-check-input {{
    width: 2.5em;
    margin-left: -2.8em;
    height: 1.25em;
}}

.form-switch .form-check-input:checked {{
    background-color: #0d6efd;
    border-color: #0d6efd;
}}
"""