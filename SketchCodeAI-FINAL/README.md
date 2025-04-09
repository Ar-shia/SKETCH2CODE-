# Sketch to Code AI

A web application that converts hand-drawn wireframe sketches into HTML/Bootstrap code using TensorFlow.js and Flask.

## Features

- Upload wireframe sketches via the web interface
- AI-powered analysis of sketches to detect UI components
- Generate HTML and CSS code based on detected components
- Support for multiple UI components (buttons, input fields, checkboxes, navbars, etc.)
- Option to use either client-side (TensorFlow.js) or server-side (TensorFlow/Keras) model

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Bootstrap, TensorFlow.js
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, CNN architectures

## Model Architecture

The application offers two different CNN model options:

1. **Transfer Learning with MobileNetV2** (default)
   - Uses a pre-trained MobileNetV2 model as the base
   - Fine-tuned for UI component detection
   - More efficient with less training data

2. **Custom CNN from Scratch**
   - Fully custom convolutional neural network
   - Four convolutional blocks with dropout and batch normalization
   - Requires more training data but potentially more customizable

## Running the Application

```bash
# Start the Flask server
python main.py
```

By default, the application uses client-side TensorFlow.js for sketch analysis. To enable the server-side CNN model, change the `USE_CNN_MODEL` flag in `app.py` to `True`.

## Training the Model

The application includes a training script for the CNN model. You can train the model using either transfer learning or from scratch:

```bash
# Train using transfer learning (default)
python train_cnn_model.py

# Train a custom CNN from scratch
python train_cnn_model.py --from-scratch

# Generate more synthetic samples
python train_cnn_model.py --samples 2000

# Customize training epochs and batch size
python train_cnn_model.py --from-scratch --epochs 150 --batch-size 8
```

The training script generates synthetic data that mimics UI component sketches. In a real-world scenario, you would use a dataset of actual UI wireframes labeled with component types and positions.

## Project Structure

- `app.py`: Main Flask application
- `main.py`: Entry point for the application
- `model/`: CNN model files
  - `cnn_model.py`: CNN architecture (MobileNetV2 or custom)
  - `preprocessing.py`: Image preprocessing functions
  - `detector.py`: UI component detector
  - `server.py`: Server interface for Flask
  - `train.py`: Training functions
- `static/`: Static files
  - `js/model.js`: Client-side CNN model
  - `js/script.js`: Frontend JavaScript
  - `css/style.css`: Custom styling
- `templates/`: HTML templates
- `train_cnn_model.py`: Standalone training script

## How It Works

1. User uploads a wireframe sketch
2. The sketch is processed (client or server-side)
3. UI components are detected with their positions and properties
4. HTML and CSS code are generated based on the detected components
5. The generated code is displayed to the user for download

## Component Types

The model can detect the following UI components:

- Buttons
- Input fields
- Checkboxes
- Navbars
- Image placeholders
- Cards
- Dropdowns
- Toggle switches