from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('plant_village_model_v1.keras')

# Class names based on PlantVillage dataset
CLASS_NAMES = ['Early_Blight', 'Late_Blight', 'Healthy']

# Image preprocessing parameters
IMG_SIZE = 256  # Adjust based on your model's input size

def preprocess_image(image_bytes):
    """Preprocess the image for model prediction"""
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Read image
        image_bytes = file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        predicted_class = CLASS_NAMES[predicted_class_index]
        
        # Get all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': class_probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Loading model...")
    print(f"Model loaded successfully!")
    print(f"Server starting on http://localhost:5000")
    app.run(debug=True, port=5000)
