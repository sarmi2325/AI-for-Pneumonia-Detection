# Pneumonia Detection from Chest X-Rays using Deep Learning

This project is a medical imaging application that detects **Pneumonia** from chest X-ray images using a deep learning model (EfficientNetB0). It includes a **Streamlit web interface**, **probability predictions**, and **Grad-CAM visualization** to highlight regions of interest on the X-ray.

---

## 🚀 Features

- 🔬 Binary classification: NORMAL vs PNEUMONIA
- 📈 Achieved 93.12% accuracy on the test dataset
- 🌐 Interactive Streamlit web app
- 📊 Displays class probabilities
- 🧯 Grad-CAM visualization for interpretability
- 📦 Clean `requirements.txt` using `pipreqs`

---

## Model Overview

- **Base Model**: `EfficientNetB0` pretrained on ImageNet
- **Fine-Tuning**: Last 5 layers unfrozen for performance boost
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Augmentation**: Brightness & contrast adjustments (light due to medical domain)

---

## Dataset

Dataset used: **Chest X-ray dataset (Kaggle)**  

## Performance

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | 93.12%    |
| Precision     | 92% (NORMAL), 95% (PNEUMONIA) |
| F1-score      | 0.93      |

Confusion Matrix:
[[278 15]
[ 25 268]]
