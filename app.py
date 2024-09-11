from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from keras_preprocessing.image import img_to_array

app = Flask(__name__)

# Load the pre-trained model and label binarizer
model = load_model('path_to_your_model.h5')
label_binarizer = pickle.load(open('label_transform.pkl', 'rb'))

default_image_size = (256, 256)

def convert_image_to_array(image):
    image = cv2.resize(image, default_image_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image = Image.open(file)
        image = np.array(image)
        
        image_array = convert_image_to_array(image)
        predictions = model.predict(image_array)
        predicted_class = label_binarizer.classes_[np.argmax(predictions)]
        
        return jsonify({'prediction': predicted_class})

    return jsonify({'error': 'Invalid file format'}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True)