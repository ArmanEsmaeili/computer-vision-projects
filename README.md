# computer-vision-projects
Computer Vision projects completed during my CV course (Grade: 20/20), covering image processing, feature detection, segmentation, object detection, and deep learning. Includes clean implementations, visual results, and performance evaluation demonstrating strong practical and theoretical understanding.

# Image Processing Fundamentals & Video Noise Analysis

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
