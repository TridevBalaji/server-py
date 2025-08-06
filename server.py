from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import io
import base64
import os

app = Flask(__name__)
CORS(app)

model = None
class_names = [
    'Abrasions', 'Bruises', 'Burns', 'Cut', 'Diabetic Wounds',
    'Laseration', 'Normal', 'Pressure Wounds', 'Surgical Wounds',
    'Venous Wounds', 'glioma', 'meningioma', 'notumor', 'pituitary'
]
def hello():
    return "Hello, World! This is the Python server."
def load_model():
    global model
    try:
        print("Loading model...")
        model_path = hf_hub_download(repo_id="tridev24/prediction", filename="trainedmodel.h5")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

def preprocess_image(image_data):
    try:
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            img = Image.open(image_data).convert('RGB')

        img = img.resize((224, 224), resample=Image.Resampling.BILINEAR)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded', 'success': False}), 500

    try:
        if 'image' not in request.files and 'imageData' not in request.json:
            return jsonify({'error': 'No image provided', 'success': False}), 400

        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected', 'success': False}), 400
            img_array = preprocess_image(file)

        elif 'imageData' in request.json:
            image_data = request.json['imageData']
            img_array = preprocess_image(image_data)

        predictions = model.predict(img_array)
        predicted_class_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_class_index]
        confidence = float(np.max(predictions))

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {'class': class_names[i], 'confidence': float(predictions[0][i])}
            for i in top_3_indices
        ]

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percentage': f"{confidence:.2%}",
            'top_predictions': top_predictions,
            'all_predictions': {
                class_names[i]: float(predictions[0][i])
                for i in range(len(class_names))
            }
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': class_names, 'total_classes': len(class_names)})

if __name__ == '__main__':
    hello()
    load_model()
    port = int(os.environ.get('PORT', 5001))
    print(f"Starting Python server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)





