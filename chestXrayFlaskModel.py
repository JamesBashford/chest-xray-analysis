from flask import Flask, request, jsonify
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask("Chest X-ray Analysis")

# Google Drive file ID (replace this with your actual file ID)
FILE_ID = '12i7ODJjsJblhbyiEFtCiUP0Ep705gBr8'

# URL for the Google Drive file
url = f'https://drive.google.com/file/d/12i7ODJjsJblhbyiEFtCiUP0Ep705gBr8/view?usp=share_link'

# Local path where the model will be saved
model_path = 'xray_tf_model.h5'

# Download the model from Google Drive
gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Pathologies list (update with your actual list)
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
