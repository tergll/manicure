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

# Load the TensorFlow Lite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.tflite')

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TensorFlow Lite model loaded successfully")


# Define labels
labels = {0: 'Needs Improvement', 1: 'Nice Manicure'}

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Process the image in-memory (no disk I/O)
    img = Image.open(file.stream).convert('RGB')  # Ensure RGB format
    img = img.resize((160, 160))  # Resize to (160, 160)
    img_array = np.array(img, dtype=np.float32) / 127.5 - 1  # Convert to [-1,1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 160, 160, 3)

    # ✅ Run inference using TensorFlow Lite
    interpreter.set_tensor(input_details[0]['index'], img_array)  # Float32 input
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # ✅ Convert output to binary class
    predicted_label = labels[int(prediction[0] > 0.5)]

    return jsonify({'prediction': predicted_label})
    print("🚀", prediction[0])

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs('uploads', exist_ok=True)
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    print("Server running on port 5000")
