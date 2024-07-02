import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import requests
from io import BytesIO

app = Flask(__name__)
model = load_model('models/my_model.keras')
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def load_and_preprocess_image_from_url(img_url):
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((32, 32))
        img_array = image.img_to_array(img)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"Error downloading or processing image from URL: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    img_url = request.json.get('imgUrl', '')
    img_array = load_and_preprocess_image_from_url(img_url)
    if img_array is not None:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        result = class_labels[predicted_class]
        return jsonify({'result': result})
    else:
        return jsonify({'result': 'Error: Failed to process the image from URL.'})

if __name__ == '__main__':
    app.run(debug=True)
