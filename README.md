# Dominant Colors Finder Web Application

A Flask-based web application that leverages machine learning algorithms to identify the dominant colors in an image. Users can upload an image from their computer or provide an image URL, and the application will display the dominant colors along with marking their presence in the image.

---

## Features

- **Upload Images**: Users can upload an image from their device.
- **Use Image URLs**: Analyze images from URLs.
- **Dominant Colors Detection**: Uses ML algorithms to detect dominant colors in the image.
- **Visual Representation**: Marks the detected dominant colors on the image.

---

## How It Works

1. **Image Input**: 
   - Upload a local image file.
   - Provide a URL of the image.

2. **Processing**:
   - The image is analyzed using machine learning algorithms to extract dominant colors.
   - The dominant colors are highlighted in the image.

3. **Output**:
   - The processed image with marked dominant colors.
   - A list of the dominant colors in HEX/RGB format.

---

## Installation and Setup

### Prerequisites

- Python 3.8+
- Flask
- NumPy, OpenCV, scikit-learn (or similar ML library)

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dominant-colors-finder.git
   cd dominant-colors-finder
