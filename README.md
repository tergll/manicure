# Gel Nail Polish Quality Classifier

_A deep learning-powered web application to classify the quality of gel nail polish application._


## 📖 Overview

This project is an **ML-driven gel nail polish quality checker** that classifies manicure quality as either:

- ✅ **"Nice Manicure"** – A well-applied gel polish.
- ❌ **"Needs Improvement"** – Uneven polish, smudging, or other flaws.

It is built using **TensorFlow, Flask, and JavaScript** and is deployed on **Render**.

---

## ⚙️ How It Works

### 🔹 **1. Model Training**
- Uses **MobileNetV2** as a feature extractor.
- Input images are **resized to `160x160` and normalized to [-1,1]**.
- The dataset consists of **good and bad manicure images**, labeled accordingly.
- **Data augmentation** helps improve generalization:
  ```python
  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip("horizontal"),
      tf.keras.layers.RandomRotation(0.3),
      tf.keras.layers.RandomZoom(0.2),
      tf.keras.layers.RandomContrast(0.2),
      tf.keras.layers.RandomBrightness(0.2)
  ])

🔹 2. Model Conversion & Deployment
The trained model is converted to TensorFlow Lite (TFLite) for efficient CPU inference.
Flask API processes images and returns predictions.
The UI is built using HTML, CSS, and JavaScript with drag-and-drop support.

🔹 3. User Interaction Flow
Upload a nail polish image via drag-and-drop or file selection.
Flask backend processes the image and runs classification.
The model returns a prediction with a confidence score.

🚀 Features
✅ Real-time classification of gel nail polish quality
✅ Fast inference with TFLite
✅ Drag-and-drop UI for easy image uploads
✅ Confidence score displayed for transparency
✅ Mobile-friendly design

📂 Folder Structure
├── static/                   # Static assets (images, styles, etc.)
│   ├── images/               # Images for UI feedback
│   ├── styles.css            # CSS file for styling
├── templates/                # HTML files
│   ├── index.html            # Main UI page
├── model/                    # ML model files
│   ├── model.tflite          # Trained TFLite model
├── app.py                    # Flask API backend
├── README.md                 # This file
├── requirements.txt          # Dependencies for Python environment
💡 Challenges & Limitations
❌ 1. Data Collection Bias
One of the biggest challenges is collecting images of bad manicures:

Social media platforms like Instagram and Pinterest mostly showcase perfect manicures.
Few people share flawed nail polish applications, making it difficult to find real-world "Needs Improvement" examples.
Solution Attempted: We manually sourced images and artificially distorted some to simulate poor applications.
❌ 2. Model Bias & Misclassification
The model may favor predicting "Nice Manicure" more often due to data imbalance.
Fix Applied: We used class weighting:
python
Copy
Edit
class_weights = {0: 1.5, 1: 1.0}  # Higher weight for "Needs Improvement"
But it's still not perfect, and further dataset expansion is needed.
❌ 3. Generalization Issues
Limited real-world testing: The model has not been tested on a diverse set of nail shapes and polish types.
Edge cases include:
Overexposed or blurry images
Unusual nail shapes
Artificially altered images (e.g., heavy filters)
🔮 Future Improvements
🔹 1. Expand Dataset with Real-World Examples
Partner with nail salons or beauty professionals to collect real images of failed manicures.
Use crowdsourcing to collect user-submitted images.
🔹 2. Fine-Tune the Model
Instead of binary classification, create a multi-class model:
🟢 "Perfect"
🟡 "Uneven Shape"
🔴 "Gel Leaked"
🔵 "Too Thick/Thin"
🔹 3. Deploy as a Mobile App
Convert the model into a TensorFlow Lite Android/iOS app.
Allow users to capture live photos and get instant feedback.
🛠️ Installation & Running Locally
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/nail-quality-checker.git
cd nail-quality-checker
2️⃣ Set Up Virtual Environment
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run Flask App
bash
Copy
Edit
python app.py
Visit http://localhost:5000/ in your browser.

