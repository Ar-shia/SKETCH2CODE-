/**
 * Sketch to Code - Main Script
 * 
 * This script handles the main functionality of the Sketch to Code application,
 * including image upload, model inference, and code generation.
 */

// Initialize variables
let sketchImage = null;
let generatedHtmlCode = '';
let generatedCssCode = '';
let detectedComponents = [];
const model = new SketchToCodeModel();
let isModelLoaded = false;

// DOM Elements
const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('sketchImage');
const imagePreview = document.getElementById('imagePreview');
const imagePreviewContainer = document.getElementById('imagePreviewContainer');
const generateBtn = document.getElementById('generateBtn');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const detectionResults = document.getElementById('detectionResults');
const detectedElementsList = document.getElementById('detectedElementsList');
const codeOutputContainer = document.getElementById('codeOutputContainer');
const generatedHtmlCodeElement = document.getElementById('generatedHtmlCode');
const generatedCssCodeElement = document.getElementById('generatedCssCode');
const copyHtmlBtn = document.getElementById('copyHtmlBtn');
const copyCssBtn = document.getElementById('copyCssBtn');
const downloadHtmlBtn = document.getElementById('downloadHtmlBtn');
const downloadCssBtn = document.getElementById('downloadCssBtn');
const saveCodeBtn = document.getElementById('saveCodeBtn');
const notificationToast = document.getElementById('notificationToast');
const toastTitle = document.getElementById('toastTitle');
const toastMessage = document.getElementById('toastMessage');
const toastIcon = document.getElementById('toastIcon');

// Initialize Bootstrap toast
const toast = new bootstrap.Toast(notificationToast);

// Event listeners
document.addEventListener('DOMContentLoaded', initializeApp);
imageInput.addEventListener('change', handleImageUpload);
generateBtn.addEventListener('click', generateCode);
copyHtmlBtn.addEventListener('click', () => copyCodeToClipboard('html'));
copyCssBtn.addEventListener('click', () => copyCodeToClipboard('css'));
downloadHtmlBtn.addEventListener('click', () => downloadCode('html'));
downloadCssBtn.addEventListener('click', () => downloadCode('css'));
saveCodeBtn.addEventListener('click', saveCodeToServer);

/**
 * Initialize the application
 */
async function initializeApp() {
    showNotification('Loading Model', 'Initializing TensorFlow.js model...', 'info');
    
    try {
        isModelLoaded = await model.loadModel();
        if (isModelLoaded) {
            showNotification('Ready', 'Model loaded successfully. Ready to process sketches!', 'success');
        } else {
            showNotification('Error', 'Failed to load model. Please refresh the page.', 'error');
        }
    } catch (error) {
        console.error('Error initializing model:', error);
        showNotification('Error', 'Failed to initialize model: ' + error.message, 'error');
    }
}

/**
 * Handle image upload and display preview
 */
function handleImageUpload(e) {
    const file = e.target.files[0];
    
    if (!file) {
        return;
    }
    
    if (!file.type.match('image.*')) {
        showNotification('Error', 'Please select an image file', 'error');
        return;
    }
    
    // Create FileReader to read the file
    const reader = new FileReader();
    
    reader.onload = function(event) {
        // Create an image element to display the sketch
        sketchImage = new Image();
        sketchImage.onload = function() {
            // Show the image preview
            imagePreview.src = this.src;
            imagePreviewContainer.classList.remove('d-none');
            generateBtn.disabled = false;
            
            // Upload the image to the server
            uploadImageToServer(file);
        };
        sketchImage.src = event.target.result;
    };
    
    reader.readAsDataURL(file);
}

/**
 * Upload image to the server
 */
function uploadImageToServer(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showNotification('Error', data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error uploading image:', error);
        showNotification('Error', 'Failed to upload image to server', 'error');
    });
}

/**
 * Generate code from the uploaded sketch
 */
async function generateCode() {
    if (!sketchImage || !isModelLoaded) {
        showNotification('Error', 'Please upload an image first or wait for the model to load', 'error');
        return;
    }
    
    // Reset previous results
    detectedComponents = [];
    generatedHtmlCode = '';
    generatedCssCode = '';
    detectedElementsList.innerHTML = '';
    
    // Show progress
    progressContainer.classList.remove('d-none');
    updateProgress(10, 'Initializing...');
    
    try {
        // Detect components in the image
        updateProgress(30, 'Analyzing sketch...');
        detectedComponents = await model.detectComponents(sketchImage);
        
        // Display detected components
        updateProgress(50, 'Identifying UI elements...');
        displayDetectedComponents(detectedComponents);
        
        // Generate HTML and CSS code
        updateProgress(70, 'Generating code...');
        generatedHtmlCode = generateHTMLCode(detectedComponents);
        generatedCssCode = generateCSSCode(detectedComponents);
        
        // Display generated code
        updateProgress(90, 'Preparing output...');
        displayGeneratedCode(generatedHtmlCode, generatedCssCode);
        
        // Complete
        updateProgress(100, 'Complete!');
        showNotification('Success', 'Code generated successfully!', 'success');
        
        // Hide progress after a delay
        setTimeout(() => {
            progressContainer.classList.add('d-none');
        }, 1000);
        
    } catch (error) {
        console.error('Error generating code:', error);
        progressContainer.classList.add('d-none');
        showNotification('Error', 'Failed to generate code: ' + error.message, 'error');
    }
}

/**
 * Update progress bar
 */
function updateProgress(percent, status) {
    progressBar.style.width = percent + '%';
    progressBar.setAttribute('aria-valuenow', percent);
    progressBar.textContent = status;
}

/**
 * Display the detected components in the UI
 */
function displayDetectedComponents(components) {
    detectionResults.classList.remove('d-none');
    detectedElementsList.innerHTML = '';
    
    components.forEach((component, index) => {
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item';
        
        const componentIcon = getComponentIcon(component.type);
        const confidenceClass = getConfidenceClass(parseFloat(component.confidence));
        
        listItem.innerHTML = `
            <div>
                <i class="${componentIcon} me-2"></i>
                <strong>${formatComponentType(component.type)}</strong>
                ${component.properties ? `<small class="text-muted">(${getComponentPropertiesSummary(component)})</small>` : ''}
            </div>
            <span class="badge ${confidenceClass}">
                ${(component.confidence * 100).toFixed(0)}%
            </span>
        `;
        
        detectedElementsList.appendChild(listItem);
    });
}

/**
 * Get appropriate icon for each component type
 */
function getComponentIcon(type) {
    switch (type) {
        case 'button': return 'fas fa-square';
        case 'input_field': return 'fas fa-keyboard';
        case 'checkbox': return 'fas fa-check-square';
        case 'navbar': return 'fas fa-bars';
        case 'image_placeholder': return 'fas fa-image';
        default: return 'fas fa-code';
    }
}

/**
 * Get confidence badge class based on confidence score
 */
function getConfidenceClass(confidence) {
    if (confidence >= 0.85) return 'bg-success';
    if (confidence >= 0.7) return 'bg-info';
    if (confidence >= 0.5) return 'bg-warning';
    return 'bg-danger';
}

/**
 * Format component type for display
 */
function formatComponentType(type) {
    return type.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
}

/**
 * Get summary of component properties
 */
function getComponentPropertiesSummary(component) {
    const props = component.properties;
    
    switch (component.type) {
        case 'button':
            return `${props.text}, ${props.variant}`;
        case 'input_field':
            return `${props.type}, ${props.label ? 'with label' : 'no label'}`;
        case 'checkbox':
            return props.label;
        case 'navbar':
            return `${props.links} links`;
        case 'image_placeholder':
            return props.aspectRatio;
        default:
            return '';
    }
}

/**
 * Generate HTML code based on detected components
 */
function generateHTMLCode(components) {
    // Start with a basic Bootstrap template
    let htmlCode = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Wireframe</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container">
`;

    // Sort components by Y position to maintain layout order
    const sortedComponents = [...components].sort((a, b) => 
        a.position.y - b.position.y
    );
    
    // Extract navbar if present
    const navbar = sortedComponents.find(c => c.type === 'navbar');
    const otherComponents = sortedComponents.filter(c => c.type !== 'navbar');
    
    // Add navbar first if present
    if (navbar) {
        htmlCode += generateNavbarCode(navbar);
    }
    
    // Add container for the rest of the components
    htmlCode += `
        <div class="row mt-4">
`;
    
    // Add other components based on their position
    otherComponents.forEach(component => {
        const colSize = determineColumnSize(component);
        htmlCode += `            <div class="${colSize} mb-3">\n`;
        htmlCode += generateComponentCode(component);
        htmlCode += `            </div>\n`;
    });
    
    // Close containers
    htmlCode += `        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>`;

    return htmlCode;
}

/**
 * Determine appropriate Bootstrap column size based on component position
 */
function determineColumnSize(component) {
    const width = component.position.width;
    
    if (width < 0.3) return 'col-md-4';
    if (width < 0.6) return 'col-md-6';
    return 'col-md-12';
}

/**
 * Generate code for a specific component
 */
function generateComponentCode(component) {
    switch (component.type) {
        case 'button':
            return generateButtonCode(component);
        case 'input_field':
            return generateInputCode(component);
        case 'checkbox':
            return generateCheckboxCode(component);
        case 'navbar':
            return generateNavbarCode(component);
        case 'image_placeholder':
            return generateImageCode(component);
        default:
            return '                <!-- Unknown component type -->\n';
    }
}

/**
 * Generate button code
 */
function generateButtonCode(component) {
    const props = component.properties;
    const variant = props.variant || 'primary';
    const size = props.size ? `btn-${props.size}` : '';
    const text = props.text || 'Button';
    
    return `                <button type="button" class="btn btn-${variant} ${size}">${text}</button>\n`;
}

/**
 * Generate input field code
 */
function generateInputCode(component) {
    const props = component.properties;
    const type = props.type || 'text';
    const placeholder = props.placeholder || '';
    const hasLabel = props.label;
    
    let code = '';
    
    if (hasLabel) {
        const labelText = placeholder || `Enter ${type}`;
        const id = `input_${Math.floor(Math.random() * 1000)}`;
        
        code += `                <div class="mb-3">
                    <label for="${id}" class="form-label">${labelText}</label>
                    <input type="${type}" class="form-control" id="${id}" placeholder="${placeholder}">
                </div>\n`;
    } else {
        code += `                <input type="${type}" class="form-control" placeholder="${placeholder}">\n`;
    }
    
    return code;
}

/**
 * Generate checkbox code
 */
function generateCheckboxCode(component) {
    const props = component.properties;
    const label = props.label || 'Checkbox';
    const checked = props.checked ? 'checked' : '';
    const id = `check_${Math.floor(Math.random() * 1000)}`;
    
    return `                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="${id}" ${checked}>
                    <label class="form-check-label" for="${id}">
                        ${label}
                    </label>
                </div>\n`;
}

/**
 * Generate navbar code
 */
function generateNavbarCode(component) {
    const props = component.properties;
    const brand = props.brand || 'Brand';
    const numLinks = props.links || 3;
    const dark = props.dark ? 'navbar-dark bg-dark' : 'navbar-light bg-light';
    
    let links = '';
    const linkNames = ['Home', 'Features', 'Pricing', 'About', 'Contact', 'Services', 'Blog'];
    
    for (let i = 0; i < Math.min(numLinks, linkNames.length); i++) {
        const active = i === 0 ? 'active' : '';
        links += `                    <li class="nav-item">
                        <a class="nav-link ${active}" href="#">${linkNames[i]}</a>
                    </li>\n`;
    }
    
    return `    <nav class="navbar navbar-expand-lg ${dark}">
        <div class="container-fluid">
            <a class="navbar-brand" href="#">${brand}</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
${links}                </ul>
            </div>
        </div>
    </nav>\n`;
}

/**
 * Generate image placeholder code
 */
function generateImageCode(component) {
    const props = component.properties;
    const aspectRatio = props.aspectRatio || '4x3';
    const border = props.border ? 'border rounded' : '';
    
    return `                <div class="ratio ratio-${aspectRatio} ${border} bg-light d-flex align-items-center justify-content-center text-muted">
                    <div>
                        <i class="fas fa-image fa-2x mb-2"></i>
                        <p>Image Placeholder</p>
                    </div>
                </div>\n`;
}

/**
 * Generate CSS code based on detected components
 */
function generateCSSCode(components) {
    // Basic CSS with custom styling for components
    let cssCode = `/* 
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
`;

    // Add specific styles for different component types
    let componentCount = {};
    
    components.forEach(component => {
        // Create unique class names for each component type
        if (!componentCount[component.type]) {
            componentCount[component.type] = 1;
        } else {
            componentCount[component.type]++;
        }
        
        // Store unique ID for this component
        if (!model.uniqueComponentIds[component.type]) {
            model.uniqueComponentIds[component.type] = 1;
        }
        const componentId = `${component.type}-${model.uniqueComponentIds[component.type]++}`;
        
        // Add component-specific CSS
        switch(component.type) {
            case 'button':
                cssCode += generateButtonCSS(component, componentId);
                break;
            case 'input_field':
                cssCode += generateInputCSS(component, componentId);
                break;
            case 'navbar':
                cssCode += generateNavbarCSS(component, componentId);
                break;
            case 'image_placeholder':
                cssCode += generateImagePlaceholderCSS(component, componentId);
                break;
        }
    });
    
    // Add responsive styling
    cssCode += `
/* Responsive styles */
@media (max-width: 768px) {
    .row > div[class^="col"] {
        margin-bottom: 1rem;
    }
    
    .navbar-brand {
        font-size: 1.2rem;
    }
}
`;
    
    return cssCode;
}

/**
 * Generate CSS for button components
 */
function generateButtonCSS(component, id) {
    const props = component.properties;
    const variant = props.variant || 'primary';
    
    return `
/* Button styles - ${id} */
.btn-${variant} {
    transition: all 0.3s ease;
}

.btn-${variant}:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
`;
}

/**
 * Generate CSS for input field components
 */
function generateInputCSS(component, id) {
    return `
/* Input field styles - ${id} */
.form-control:focus {
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
    border-color: #80bdff;
}

.form-label {
    font-weight: 500;
    margin-bottom: 0.5rem;
}
`;
}

/**
 * Generate CSS for navbar components
 */
function generateNavbarCSS(component, id) {
    const props = component.properties;
    const dark = props.dark;
    
    return `
/* Navbar styles - ${id} */
.navbar {
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 1.5rem;
}

.navbar-brand {
    font-weight: bold;
    letter-spacing: 0.5px;
}

.nav-link {
    position: relative;
}

.nav-link.active:after {
    content: '';
    position: absolute;
    left: 0;
    bottom: -2px;
    width: 100%;
    height: 2px;
    background-color: ${dark ? '#fff' : '#007bff'};
}
`;
}

/**
 * Generate CSS for image placeholder components
 */
function generateImagePlaceholderCSS(component, id) {
    return `
/* Image placeholder styles - ${id} */
.ratio {
    transition: all 0.3s ease;
    overflow: hidden;
}

.ratio:hover {
    opacity: 0.9;
    cursor: pointer;
}

.ratio i, .ratio p {
    opacity: 0.7;
}
`;
}

/**
 * Display the generated code
 */
function displayGeneratedCode(htmlCode, cssCode) {
    // Display HTML code with syntax highlighting
    generatedHtmlCodeElement.textContent = htmlCode;
    Prism.highlightElement(generatedHtmlCodeElement);
    
    // Display CSS code with syntax highlighting
    generatedCssCodeElement.textContent = cssCode;
    Prism.highlightElement(generatedCssCodeElement);
    
    // Show code output container
    codeOutputContainer.classList.remove('d-none');
}

/**
 * Copy generated code to clipboard
 */
function copyCodeToClipboard(type) {
    let codeToCopy = '';
    
    if (type === 'html') {
        if (!generatedHtmlCode) {
            showNotification('Error', 'No HTML code to copy!', 'error');
            return;
        }
        codeToCopy = generatedHtmlCode;
    } else if (type === 'css') {
        if (!generatedCssCode) {
            showNotification('Error', 'No CSS code to copy!', 'error');
            return;
        }
        codeToCopy = generatedCssCode;
    } else {
        showNotification('Error', 'Invalid code type specified!', 'error');
        return;
    }
    
    navigator.clipboard.writeText(codeToCopy)
        .then(() => {
            showNotification('Success', `${type.toUpperCase()} code copied to clipboard!`, 'success');
        })
        .catch(error => {
            console.error('Error copying to clipboard:', error);
            showNotification('Error', 'Failed to copy code: ' + error.message, 'error');
        });
}

/**
 * Download the generated code as HTML or CSS file
 */
function downloadCode(type) {
    let codeToDownload = '';
    let fileName = '';
    let mimeType = '';
    
    if (type === 'html') {
        if (!generatedHtmlCode) {
            showNotification('Error', 'No HTML code to download!', 'error');
            return;
        }
        codeToDownload = generatedHtmlCode;
        fileName = 'generated_wireframe.html';
        mimeType = 'text/html';
    } else if (type === 'css') {
        if (!generatedCssCode) {
            showNotification('Error', 'No CSS code to download!', 'error');
            return;
        }
        codeToDownload = generatedCssCode;
        fileName = 'generated_wireframe.css';
        mimeType = 'text/css';
    } else {
        showNotification('Error', 'Invalid code type specified!', 'error');
        return;
    }
    
    const blob = new Blob([codeToDownload], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
    
    showNotification('Success', `${type.toUpperCase()} code downloaded successfully!`, 'success');
}

/**
 * Save the generated code to the server
 */
function saveCodeToServer() {
    if (!generatedHtmlCode) {
        showNotification('Error', 'No HTML code to save!', 'error');
        return;
    }
    
    // Save both HTML and CSS files
    fetch('/save-code', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            html_code: generatedHtmlCode,
            css_code: generatedCssCode || ''
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showNotification('Error', data.error, 'error');
        } else {
            showNotification('Success', 'Code saved to server successfully!', 'success');
        }
    })
    .catch(error => {
        console.error('Error saving code:', error);
        showNotification('Error', 'Failed to save code to server: ' + error.message, 'error');
    });
}

/**
 * Show a notification toast
 */
function showNotification(title, message, type) {
    // Set toast content
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    
    // Set icon based on notification type
    toastIcon.className = 'me-2 fas ';
    switch (type) {
        case 'success':
            toastIcon.className += 'fa-check-circle text-success';
            break;
        case 'error':
            toastIcon.className += 'fa-exclamation-circle text-danger';
            break;
        case 'warning':
            toastIcon.className += 'fa-exclamation-triangle text-warning';
            break;
        case 'info':
        default:
            toastIcon.className += 'fa-info-circle text-info';
    }
    
    // Show the toast
    toast.show();
}
