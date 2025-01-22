from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
from PIL import Image
import tensorflow as tf
# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Create the Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)
    print('Model loaded')
    print(model)
    print("Is TensorFlow using GPU?", tf.test.is_gpu_available())


# Define labels
labels = {0: 'Needs Improvement', 1: 'Nice Manicure'}

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Save the file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    # Preprocess the image
    # Preprocess the image
    img = Image.open(file_path).convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((160, 160))  # Resize to (160, 160)
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 160, 160, 3)

    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = labels[int(prediction[0])]
    
    # Remove the file after prediction
    os.remove(file_path)
    
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs('uploads', exist_ok=True)
    # app.run(debug=True)
    # import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    print("Server running on port 5000")
