import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request

# Function to download and preprocess the image from URL
def load_and_preprocess_image_from_url(img_url):
    # Download the image from URL
    try:
        with urllib.request.urlopen(img_url) as url_response:
            original_img = Image.open(url_response)
            original_img = original_img.convert('RGB')  # Ensure it's in RGB format
            img = original_img.resize((32, 32))  # Resize to 32x32 pixels
            img_array = image.img_to_array(img)
            
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
            return img_array, original_img, img  # Return preprocessed image array, original PIL image, and resized image
    except Exception as e:
        print(f"Error downloading or processing image from URL: {e}")
        return None, None, None

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Image Recognition Script')
parser.add_argument('--url', type=str, help='URL of the image to recognize')
args = parser.parse_args()

# Load the trained model
model = load_model('models/my_model.keras')  # Replace with your actual model file path

# Class labels for CIFAR-10
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

if args.url:
    # Predict on the image from URL provided via command-line argument
    img_url = args.url
    img_array, original_img, resized_img = load_and_preprocess_image_from_url(img_url)

    if img_array is not None:
        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Print predicted class
        print(f"Image from URL '{img_url}': Predicted class: {class_labels[predicted_class]}")

        # Display the original and resized images with the prediction result
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axs[0].imshow(original_img)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        # Resized image with prediction
        axs[1].imshow(resized_img)
        axs[1].set_title(f"Resized Image\nPredicted: {class_labels[predicted_class]}")
        axs[1].axis('off')
        
        plt.show()
    else:
        print(f"Error: Failed to download or process the image from URL '{img_url}'")
else:
    print("Please provide a valid URL using the '--url' argument.")
