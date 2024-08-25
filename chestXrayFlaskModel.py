from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask("Chest X-ray Analysis")

# Load the model
model = tf.keras.models.load_model('/Users/jamesbashford/Downloads/xray_tf_model-2.h5')

# Pathologies list
pathologies = [
    'No Finding',
    'Infiltration',
    'Nodule',
    'Fibrosis',
    'Pleural_Thickening',
    'Atelectasis',
    'Hernia',
    'Effusion',
    'Mass',
    'Cardiomegaly',
    'Pneumothorax',
    'Emphysema',
    'Consolidation',
    'Edema',
    'Pneumonia'
]

# Function to preprocess an image
def preprocess_image(image):
    img = Image.open(image).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_array = preprocess_image(file)
    predictions = model.predict(img_array)
    sorted_indices = np.argsort(predictions[0])[::-1]
    sorted_pathologies = [(pathologies[i], predictions[0][i]) for i in sorted_indices]

    return jsonify(sorted_pathologies)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

