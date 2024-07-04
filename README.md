# Image Recognition App

This is a Flask web application for image recognition using a pre-trained Keras model on the CIFAR-10 dataset. It allows users to classify objects in images by entering an image URL.

## Features

- **Classification**: Upload an image URL to classify objects into categories such as Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.
- **Display**: Show both the resized (32x32) and original images alongside the classification result.
- **Responsive**: Designed with a responsive layout for usability on different devices.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ape-with-helmet/Image-recognition-CIFAR10-dataset.git
   cd Image-recognition-CIFAR10-dataset
   ```

2. Install dependencies using pip (recommended to run on a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```
   The application will run locally at `http://localhost:5000`.

## Usage

- Open your web browser and navigate to `http://localhost:5000`.
- Enter the URL of an image you want to classify.
- Click the "Classify" button to see the classification result and images.

## About

This project uses a TensorFlow/Keras model trained on the CIFAR-10 dataset for image classification. It preprocesses images by resizing them to 32x32 pixels before classification. The Flask framework is used to create a web interface where users can interact with the model by providing image URLs.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
