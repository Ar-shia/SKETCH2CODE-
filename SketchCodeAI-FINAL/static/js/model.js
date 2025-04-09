/**
 * Model.js
 * 
 * This file contains the TensorFlow.js model implementation for the
 * Sketch to Code converter. It simulates a CNN model trained with a Hybrid 
 * Evolutionary Algorithm to detect UI components from sketches.
 */

class SketchToCodeModel {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.componentClasses = [
            'button', 
            'input_field', 
            'checkbox',
            'navbar',
            'image_placeholder'
        ];
        this.uniqueComponentIds = {}; // To track unique IDs for components
    }

    /**
     * Initialize and load the TensorFlow.js model
     */
    async loadModel() {
        try {
            // In a real application, we would load a pre-trained model here
            // For this demo, we'll create a simple model that simulates component detection
            
            // Create a simple convolutional model for demonstration
            this.model = tf.sequential();
            
            // Add layers to the model
            this.model.add(tf.layers.conv2d({
                inputShape: [224, 224, 3],
                filters: 16,
                kernelSize: 3,
                activation: 'relu'
            }));
            
            this.model.add(tf.layers.maxPooling2d({
                poolSize: [2, 2]
            }));
            
            this.model.add(tf.layers.conv2d({
                filters: 32,
                kernelSize: 3,
                activation: 'relu'
            }));
            
            this.model.add(tf.layers.maxPooling2d({
                poolSize: [2, 2]
            }));
            
            this.model.add(tf.layers.flatten());
            
            this.model.add(tf.layers.dense({
                units: 64,
                activation: 'relu'
            }));
            
            this.model.add(tf.layers.dense({
                units: this.componentClasses.length,
                activation: 'softmax'
            }));
            
            // Compile the model
            this.model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            this.isModelLoaded = true;
            console.log("SketchToCode model initialized");
            
            return true;
        } catch (error) {
            console.error("Error loading model:", error);
            return false;
        }
    }

    /**
     * Preprocess the image for the model
     * @param {HTMLImageElement} image - The image element to process
     * @returns {tf.Tensor} The processed image tensor
     */
    preprocessImage(image) {
        return tf.tidy(() => {
            // Convert image to tensor
            const imageTensor = tf.browser.fromPixels(image);
            
            // Resize image to model input size
            const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);
            
            // Normalize pixel values to [0, 1]
            const normalizedImage = resizedImage.div(tf.scalar(255));
            
            // Expand dimensions to match model input shape [1, 224, 224, 3]
            return normalizedImage.expandDims(0);
        });
    }

    /**
     * Simulate component detection in different regions of the image
     * @param {HTMLImageElement} image - The image to analyze
     * @returns {Promise<Array>} Detected components with positions and confidence scores
     */
    async detectComponents(image) {
        if (!this.isModelLoaded) {
            await this.loadModel();
        }
        
        // In a real implementation, we would:
        // 1. Preprocess the image
        // 2. Run object detection to find bounding boxes of UI components
        // 3. Classify each detected region
        
        // For this demo, we'll simulate detection results
        
        // Analyze image for detecting components
        const detectedComponents = await this.simulateDetection(image);
        
        return detectedComponents;
    }

    /**
     * Simulate the detection process
     * @param {HTMLImageElement} image - The source image
     * @returns {Promise<Array>} Array of detected UI components with properties
     */
    async simulateDetection(image) {
        // Get image dimensions for relative positioning
        const width = image.width;
        const height = image.height;
        
        // Create a canvas to analyze the image
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, width, height);
        
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Analyze different regions of the image to simulate detection
        const imageData = ctx.getImageData(0, 0, width, height);
        
        // For demo purposes, we'll create a deterministic pattern of components
        // based on image analysis
        
        // Divide the image into a grid for analysis
        const gridSize = 3;
        const cellWidth = width / gridSize;
        const cellHeight = height / gridSize;
        
        const components = [];
        
        // Calculate brightness in each cell to determine component type
        for (let y = 0; y < gridSize; y++) {
            for (let x = 0; x < gridSize; x++) {
                const cellX = Math.floor(x * cellWidth);
                const cellY = Math.floor(y * cellHeight);
                
                // Skip some cells randomly to make the detection more realistic
                if (Math.random() > 0.7) continue;
                
                // Analyze brightness in this region to determine component
                const brightness = this.calculateRegionBrightness(
                    imageData, 
                    cellX, 
                    cellY, 
                    Math.floor(cellWidth), 
                    Math.floor(cellHeight)
                );
                
                // Map brightness to component type and confidence
                const componentIndex = Math.floor((brightness / 255) * this.componentClasses.length);
                const componentType = this.componentClasses[
                    Math.min(componentIndex, this.componentClasses.length - 1)
                ];
                
                // Add some randomness to make it more realistic
                const confidence = 0.7 + (Math.random() * 0.25);
                
                components.push({
                    type: componentType,
                    confidence: confidence.toFixed(2),
                    position: {
                        x: cellX / width,
                        y: cellY / height,
                        width: cellWidth / width,
                        height: cellHeight / height
                    },
                    properties: this.generateComponentProperties(componentType)
                });
            }
        }
        
        // Add a navbar at the top if none exists
        if (!components.some(c => c.type === 'navbar')) {
            components.push({
                type: 'navbar',
                confidence: 0.85,
                position: {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 0.1
                },
                properties: this.generateComponentProperties('navbar')
            });
        }
        
        return components;
    }

    /**
     * Calculate average brightness in an image region
     */
    calculateRegionBrightness(imageData, x, y, width, height) {
        let totalBrightness = 0;
        let pixelCount = 0;
        
        for (let j = y; j < y + height && j < imageData.height; j++) {
            for (let i = x; i < x + width && i < imageData.width; i++) {
                const index = (j * imageData.width + i) * 4;
                const r = imageData.data[index];
                const g = imageData.data[index + 1];
                const b = imageData.data[index + 2];
                
                // Calculate pixel brightness
                const brightness = (r + g + b) / 3;
                totalBrightness += brightness;
                pixelCount++;
            }
        }
        
        return pixelCount > 0 ? totalBrightness / pixelCount : 0;
    }

    /**
     * Generate specific properties for each component type
     */
    generateComponentProperties(type) {
        switch (type) {
            case 'button':
                return {
                    text: this.getRandomButtonText(),
                    variant: this.getRandomButtonVariant(),
                    size: this.getRandomFromArray(['sm', 'md', 'lg'])
                };
                
            case 'input_field':
                return {
                    placeholder: this.getRandomPlaceholder(),
                    type: this.getRandomFromArray(['text', 'email', 'password', 'number']),
                    label: Math.random() > 0.5
                };
                
            case 'checkbox':
                return {
                    label: this.getRandomCheckboxLabel(),
                    checked: Math.random() > 0.5
                };
                
            case 'navbar':
                return {
                    brand: 'Brand Name',
                    links: Math.floor(2 + Math.random() * 4),
                    dark: true
                };
                
            case 'image_placeholder':
                return {
                    aspectRatio: this.getRandomFromArray(['1x1', '4x3', '16x9']),
                    border: Math.random() > 0.5
                };
                
            default:
                return {};
        }
    }

    /**
     * Helper functions to generate random component properties
     */
    getRandomButtonText() {
        const texts = ['Submit', 'Save', 'Cancel', 'Send', 'Login', 'Register', 'Download', 'Upload', 'Next', 'Previous'];
        return this.getRandomFromArray(texts);
    }
    
    getRandomButtonVariant() {
        const variants = ['primary', 'secondary', 'success', 'danger', 'warning', 'info', 'light', 'dark'];
        return this.getRandomFromArray(variants);
    }
    
    getRandomPlaceholder() {
        const placeholders = ['Enter text...', 'Email address', 'Password', 'Search...', 'Your name', 'Phone number'];
        return this.getRandomFromArray(placeholders);
    }
    
    getRandomCheckboxLabel() {
        const labels = ['Remember me', 'I agree to terms', 'Subscribe to newsletter', 'Save my information'];
        return this.getRandomFromArray(labels);
    }
    
    getRandomFromArray(array) {
        return array[Math.floor(Math.random() * array.length)];
    }
}

// Export the model class
window.SketchToCodeModel = SketchToCodeModel;
