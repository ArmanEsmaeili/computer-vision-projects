# computer-vision-projects
Computer Vision projects completed during my CV course (Grade: 20/20), covering image processing, feature detection, segmentation, object detection, and deep learning. Includes clean implementations, visual results, and performance evaluation demonstrating strong practical and theoretical understanding.

# 01_image_processing_fundamentals

## Image Processing Fundamentals & Video Noise Analysis

This project covers foundational concepts in computer vision and image processing, implemented using Python and OpenCV. The objective was to explore color models, contrast analysis, histogram processing, spatial filtering, edge detection, and basic video processing.

---

## Objectives

- Understand image color representations (RGB, Grayscale, Binary, HSV)
- Analyze and manipulate image contrast and histogram distributions
- Study noise models and spatial filtering techniques
- Implement edge detection algorithms
- Perform basic video frame processing

---

## Part 1: Color Models & Histogram Analysis

### Color Space Conversion
- Loaded image and analyzed:
  - Data type
  - Number of channels
- Converted image to:
  - Grayscale
  - Binary
  - HSV
- Extracted and visualized individual channels (R, G, B, H, S, V)

### Contrast Analysis
- Defined contrast mathematically
- Increased and decreased contrast
- Compared results visually

### Histogram Processing
- Plotted image histogram
- Applied:
  - Histogram Equalization
  - Histogram Stretching
- Analyzed intensity redistribution effects

---

## Part 2: Noise & Spatial Filtering

### Noise Injection
- Added Salt & Pepper noise
- Discussed Gaussian noise as a comparative model

### Spatial Filters Applied
- Mean Filter
- Gaussian Filter
- Median Filter

Analyzed:
- Kernel structure
- Effect on noise removal
- Trade-off between smoothing and detail preservation

Median filter was identified as the most effective for Salt & Pepper noise removal.

### Edge Detection
Implemented and analyzed:
- Sobel Operator
- Laplacian Operator
- Canny Edge Detector

Compared gradient-based vs multi-stage edge detection approaches.

---

## Part 3: Video Processing

- Loaded video file
- Extracted:
  - Number of frames
  - FPS
- Added Salt & Pepper noise to each frame
- Generated processed output video

---

## Skills Demonstrated

- OpenCV image manipulation
- Color space theory
- Histogram analysis
- Convolution and kernel operations
- Noise modeling
- Edge detection algorithms
- Frame-wise video processing
- Analytical comparison of filtering methods

---

## Tools Used

- Python
- OpenCV
- NumPy
- Matplotlib

# 02_captcha_generation_and_processing

## CAPTCHA Generation and Image Processing

This project focuses on the generation and analysis of CAPTCHA images using Python-based image processing techniques. The objective was to simulate real-world CAPTCHA systems and explore how noise, distortion, and visual transformations affect text readability and automated recognition.

---

## Project Overview

- Programmatically generated CAPTCHA images with random text
- Applied image distortions and transformations
- Introduced controlled noise patterns
- Performed preprocessing techniques to enhance readability
- Analyzed the impact of filtering and thresholding methods

---

## Technical Concepts Covered

- Image rendering and text overlay
- Noise modeling and augmentation
- Geometric distortions
- Thresholding and binarization
- Morphological operations
- Preprocessing pipelines for OCR readiness

---

## Skills Demonstrated

- Synthetic dataset generation
- Practical noise simulation
- Image preprocessing design
- Understanding of robustness in vision systems
- Clean implementation of processing pipelines

---

# 03_Pretrained_CV_Models

## Project Overview
This project focuses on the **practical use of pre-trained models in computer vision**. The main goal is to understand how pre-trained models can be applied to different computer vision tasks without additional training.  

The project covers three main tasks:  
1. **Image Classification** – Predicting the class of input images.  
2. **Object Detection** – Detecting and localizing objects within images.  
3. **Semantic Segmentation** – Generating pixel-level masks to segment images based on object categories.  

All tasks are applied on a variety of input images including urban scenes, nature, and other real-world scenarios.

---

## Project Steps

### 1. Image Classification
- Load a pre-trained model in inference mode.  
- Predict the class for each input image and output the **top class with confidence score**.  
- Analyze which predictions are correct and where the model fails.  
- Understand how pre-trained models generalize to unseen images.

### 2. Object Detection
- Complete the provided code to draw **bounding boxes** on detected objects.  
- Display **class names and confidence scores** on the image.  
- Analyze the detection results for accuracy and potential errors.

### 3. Semantic Segmentation
- Apply a pre-trained segmentation model to generate **color-coded masks** based on the VOC color palette.  
- Overlay the mask on the original image to visualize segmentation.  
- Measure and analyze the proportion of each mask over the original image to evaluate performance.

---

## Key Objectives
- Learn how to **load and use pre-trained model weights** without training from scratch.  
- Evaluate model performance across classification, detection, and segmentation tasks.  
- Understand the limitations and strengths of pre-trained models in real-world applications.  

---

## Technologies and Libraries
- **Python**  
- Deep learning frameworks: **PyTorch** or **TensorFlow**  
- **OpenCV** for image processing  
- **Matplotlib** for visualization
## Tools Used

- Python
- OpenCV
- NumPy
- Matplotlib

- # 03_FashionMNIST_Classification

## Project Overview
This project focuses on designing, training, and evaluating a **multi-class neural network** using **PyTorch** on the **Fashion-MNIST dataset**. The aim is to understand the full workflow of a neural network, including:

- Loading and preprocessing image data  
- Building a multi-layer neural network  
- Training and evaluating the model  
- Analyzing results and improving performance  

The Fashion-MNIST dataset contains **60,000 training images** and **10,000 test images**, each **28x28 grayscale**, belonging to **10 clothing categories**, including t-shirts, coats, shoes, boots, bags, and more.

---

## Project Steps

### 1. Data Loading and Exploration
- Load Fashion-MNIST using `torchvision.datasets.FashionMNIST`.  
- Display sample images with labels to understand dataset structure.  
- Flatten each image to 1D vectors (28x28 → 784) for input to the network.  
- Examine batch shapes and understand why flattening is necessary for fully connected layers.

### 2. Model Design
- Build a **multi-layer fully connected neural network** as a subclass of `nn.Module`:  
  - Input layer: 784 → 256, **ReLU activation**  
  - Hidden layer: 256 → 128, **ReLU activation**  
  - Output layer: 128 → 10, raw outputs (logits)  
- Print model structure and total parameters.  
- Understand the role of each layer and how parameters contribute to learning.

### 3. Loss Function and Optimizer
- Use **CrossEntropyLoss** for multi-class classification.  
- Use **Adam optimizer** with learning rate `0.001`.  
- Brief explanation: CrossEntropyLoss is suitable for multi-class tasks as it combines softmax and negative log-likelihood in one step.

### 4. Training the Model
- Implement the full training loop:  
  - Set model to `train()` mode  
  - Forward pass through batches  
  - Compute loss, perform `backward()`  
  - Update weights using optimizer  
  - Track loss and accuracy per epoch  
- Recommended epochs: 10+  
- Print accuracy after each epoch for monitoring.

### 5. Model Evaluation
- Switch model to `eval()` mode for testing.  
- Compute accuracy on test dataset.  
- Display some predicted images with actual labels.  
- Analyze misclassified examples to understand model weaknesses.

### 6. Visualization
- Plot **loss vs. epochs**  
- Plot **accuracy vs. epochs**  
- Analyze potential overfitting or underfitting trends.

### 7. Optional Model Improvements
- Add **Dropout** to reduce overfitting  
- Add **Batch Normalization** for faster convergence  
- Experiment with learning rates, optimizer types, or number of neurons  
- Compare results to the base model and analyze improvements

---

## Technologies and Libraries
- **Python**  
- **PyTorch** for neural network modeling  
- **Torchvision** for dataset loading  
- **NumPy** for array operations  
- **Matplotlib** for visualization  

---

## Key Objectives
- Understand the workflow of building a **multi-class classification neural network**  
- Learn to preprocess and flatten image data for neural networks  
- Train and evaluate a neural network with **PyTorch**  
- Explore methods to improve model performance (Dropout, BatchNorm, learning rate tuning)
