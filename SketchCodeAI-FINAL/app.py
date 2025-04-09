import os
import logging
from flask import Flask, render_template, request, jsonify
import base64

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "sketchtocodedefaultsecret")

# Configuration flags
USE_CNN_MODEL = False  # Set to True to use the server-side CNN model
USE_CUSTOM_CNN = True  # Set to True for custom CNN, False for MobileNetV2 transfer learning

# If the CNN model is enabled, we'll import it here
model_initialized = False
if USE_CNN_MODEL:
    try:
        import numpy as np
        import tensorflow as tf
        from model.server import initialize_model, process_image, process_image_file
        
        # Configure TensorFlow
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.error(f"Error configuring GPUs: {str(e)}")
        
        # Initialize the model with the selected architecture
        model_type = "custom CNN (from scratch)" if USE_CUSTOM_CNN else "MobileNetV2 (transfer learning)"
        logger.info(f"Initializing CNN model using {model_type}...")
        model_initialized = initialize_model(from_scratch=USE_CUSTOM_CNN)
        logger.info(f"Model initialization {'successful' if model_initialized else 'failed'}")
    except Exception as e:
        logger.error(f"Error initializing CNN model: {str(e)}")
        USE_CNN_MODEL = False
        model_initialized = False
else:
    logger.info("CNN model is disabled")
    model_initialized = False
    
    # Define a placeholder for process_image_file when CNN is disabled
    def process_image_file(file_path):
        logger.warning("CNN model is disabled, but process_image_file was called")
        return False, {"error": "CNN model is disabled"}

@app.route('/')
def index():
    """Render the main page"""
    # Determine model configuration for display
    model_info = {
        'enabled': USE_CNN_MODEL,
        'architecture': 'Custom CNN (from scratch)' if USE_CUSTOM_CNN else 'MobileNetV2 (transfer learning)',
        'initialized': model_initialized
    }
    return render_template('index.html', model_info=model_info)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and return the image data to be processed by TensorFlow.js"""
    try:
        # Get the image from the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Generate a unique filename
        import uuid
        import datetime
        
        filename = f"sketch_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}{os.path.splitext(image_file.filename)[1]}"
        file_path = os.path.join(app.static_folder, 'uploads', filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        image_file.save(file_path)
        
        # Read and encode the image
        with open(file_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # If CNN model is enabled and initialized, process with it
        if USE_CNN_MODEL and model_initialized:
            try:
                logger.info(f"Processing image with CNN model: {file_path}")
                success, result = process_image_file(file_path)
                
                if success:
                    logger.info(f"Successfully detected {len(result.get('components', []))} components")
                    # Return the encoded image data and detection results
                    return jsonify({
                        'success': True,
                        'image_data': image_data,
                        'image_url': f'/static/uploads/{filename}',
                        'use_server_detection': True,
                        'components': result.get('components', []),
                        'html_code': result.get('html_code', ''),
                        'css_code': result.get('css_code', '')
                    })
                else:
                    logger.warning(f"Model detection failed: {result.get('error', 'Unknown error')}")
                    # Fall back to client-side processing
                    return jsonify({
                        'success': True,
                        'image_data': image_data,
                        'image_url': f'/static/uploads/{filename}',
                        'use_server_detection': False,
                        'error': result.get('error', 'Component detection failed')
                    })
            except Exception as e:
                logger.error(f"Error in CNN model processing: {str(e)}")
                # Fall back to client-side processing
        
        # Use client-side processing
        logger.info("Using client-side processing for sketch recognition")
        return jsonify({
            'success': True,
            'image_data': image_data,
            'image_url': f'/static/uploads/{filename}',
            'use_server_detection': False
        })
    
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/save-code', methods=['POST'])
def save_code():
    """Save the generated code to files on the server"""
    try:
        data = request.json
        html_code = data.get('html_code')
        css_code = data.get('css_code')
        
        if not html_code:
            return jsonify({'error': 'No HTML code provided'}), 400
        
        # Generate unique filenames
        import uuid
        import datetime
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        
        html_filename = f"sketch_html_{timestamp}_{unique_id}.html"
        css_filename = f"sketch_css_{timestamp}_{unique_id}.css"
        
        html_file_path = os.path.join(app.static_folder, 'generated', html_filename)
        css_file_path = os.path.join(app.static_folder, 'generated', css_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
        
        # Write the HTML code to the file
        with open(html_file_path, 'w') as f:
            f.write(html_code)
        
        # Write the CSS code to file if provided
        css_file_url = None
        if css_code:
            with open(css_file_path, 'w') as f:
                f.write(css_code)
            css_file_url = f'/static/generated/{css_filename}'
        
        # Return the paths to the saved files
        return jsonify({
            'success': True,
            'html_file_path': f'/static/generated/{html_filename}',
            'css_file_path': css_file_url
        })
    
    except Exception as e:
        app.logger.error(f"Error saving code: {str(e)}")
        return jsonify({'error': f'Error saving code: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
