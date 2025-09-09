# Gel Nail Polish Quality Classifier

_A deep learning-powered web application to classify the quality of gel nail polish application._

---

## Overview

This project is an **ML-driven gel nail polish quality checker** that classifies manicure quality as either:

- **"Nice Manicure"** – A well-applied gel polish.  
- **"Needs Improvement"** – Uneven polish, smudging, or other flaws.

It is built using **TensorFlow, Flask, and JavaScript**, and deployed on **Render**.

---

## How It Works

### 1. Model Training
- Uses **MobileNetV2** as a feature extractor.  
- Input images are **resized to `160x160` and normalized to [-1, 1]**.  
- Dataset consists of **good and bad manicure images**, labeled accordingly.  
- **Data augmentation** improves generalization:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2)
])
```

### 2. Model Conversion & Deployment
- The trained model is converted to **TensorFlow Lite (TFLite)** for efficient CPU inference.  
- A **Flask API** processes images and returns predictions.  
- The **UI** is built with HTML, CSS, and JavaScript, featuring drag-and-drop support.  

### 3. User Interaction Flow
1. Upload a nail polish image via drag-and-drop or file selection.  
2. The Flask backend processes the image and runs classification.  
3. The model returns a prediction with a confidence score.  

---

## Features
- Real-time classification of gel nail polish quality  
- Fast inference with TensorFlow Lite  
- Drag-and-drop UI for easy image uploads  
- Confidence score displayed for transparency  
- Mobile-friendly design  

---

## Folder Structure

```
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
```

---

## Challenges & Limitations

### 1. Data Collection Bias
Collecting images of poor manicures is difficult:
- Social media platforms mostly showcase perfect manicures.  
- Few people share flawed nail polish applications.  

**Solution Attempted:** Manually sourced images and artificially distorted some to simulate poor applications.

### 2. Model Bias & Misclassification
- The model may favor predicting "Nice Manicure" due to dataset imbalance.  
- **Fix Applied:** Class weighting:  

```python
class_weights = {0: 1.5, 1: 1.0}  # Higher weight for "Needs Improvement"
```

Still, further dataset expansion is needed.

### 3. Generalization Issues
The model has not been tested on a wide variety of conditions:  
- Overexposed or blurry images  
- Unusual nail shapes  
- Artificially altered images (filters, editing, etc.)  

---

## Future Improvements

1. **Expand Dataset with Real-World Examples**  
   - Partner with nail salons or beauty professionals to collect real-world images.  
   - Use crowdsourcing for user-submitted images.  

2. **Fine-Tune the Model**  
   - Extend from binary classification to multi-class categories, e.g.:  
     - "Perfect"  
     - "Uneven Shape"  
     - "Gel Leaked"  
     - "Too Thick/Thin"  

3. **Deploy as a Mobile App**  
   - Convert to TensorFlow Lite for Android/iOS.  
   - Enable live photo capture with instant feedback.  

---

## Installation & Running Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/nail-quality-checker.git
   cd nail-quality-checker
   ```

2. **Set Up Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Flask App**
   ```bash
   python app.py
   ```

Visit `http://localhost:5000/` in your browser.
