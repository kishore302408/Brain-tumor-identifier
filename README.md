# 🧠 NeuroScan Pro — Brain Tumor MRI Classification

**MED-02 Medical AI Challenge** · Brain Tumor Classification / Medical Image Analysis from MRI Scans

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Flask](https://img.shields.io/badge/API-Flask-000000)
![PyTorch](https://img.shields.io/badge/Model-PyTorch-EE4C2C)
![Render](https://img.shields.io/badge/Deploy-Render-46E3B7)

**🚀 Live Demo: [https://brain-tumor-identifier-1.onrender.com/](https://brain-tumor-identifier-1.onrender.com/)**

---

## Overview

NeuroScan Pro is a production-ready web AI tool that classifies brain MRI scans into four classes — **glioma, meningioma, no tumor, pituitary** — with confidence scores, Grad-CAM explainability, parametric imaging features, and clinical performance metrics.

Built for the MED-02 challenge: upload an MRI → get an AI prediction with an explainable heatmap in seconds.

## ✨ Features

| Feature | Description |
|---|---|
| 📤 **Multi-image upload** | Analyze a batch of MRI scans in one run |
| 🤖 **3 trained models** | CNN, ResNet18, MobileNetV2 — pick one or run the ensemble |
| 🔥 **Grad-CAM heatmaps** | Visualise exactly which regions drove the prediction |
| 🎯 **Confidence scores** | Per-class probability breakdown for every scan |
| 📊 **Parametric features** | Intensity/texture statistics of the scan |
| 🟡 **Tumour localisation mask** | Approximate segmentation overlay toggle |
| 📈 **Performance dashboard** | Accuracy, confusion matrix, precision/recall/F1 per model |
| ⚖️ **Comparison table + CSV export** | Side-by-side results across models, downloadable report |
| 🌙 **Professional dark UI** | Medical-grade interface built with Streamlit |

## 🏗️ Architecture

```
Brain-tumor-identifier/
├── streamlit_app.py     # Web UI (Grad-CAM, metrics, comparison, export)
├── app.py               # Flask REST API (/api/predict, /api/health)
├── nets.py              # CNN / ResNet18 / MobileNetV2 architectures + target layers
├── train.py             # Training + evaluation pipeline
├── weights/             # Trained model weights (*.pt)
├── metrics/             # Accuracy, classification reports, confusion matrices (*.json)
├── static/index.html    # Flask API landing page
├── render.yaml           # Render deployment blueprint
└── requirements.txt      # Dependencies
```

**Stack:** PyTorch · Torchvision · Streamlit · Flask · pandas · scikit-learn · Grad-CAM

## 📊 Results

Trained on a 4-class brain MRI dataset (train/test split, 80/20):

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **ResNet18** ⭐ | **95.13%** | 0.951 | Best model — default in the app |
| **MobileNetV2** | **95.00%** | 0.949 | Lightweight, fast |
| **CNN (baseline)** | **80.13%** | 0.795 | Custom 4-layer CNN |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain on your own dataset
python train.py path/to/dataset 8

# 3a. Run the web UI
python -m streamlit run streamlit_app.py --server.port 8502

# 3b. Or run the REST API
python app.py   # Flask on port 5001
```

## ☁️ Deployment (Render)

The app is deployed on Render via the included `render.yaml` blueprint:

```yaml
startCommand: python -m streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

- Models are **lazily loaded** on first analysis for fast cold starts on the free tier.
- Every push to this repository auto-redeploys the live site.

## 📂 Dataset Format

```
dataset/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

## ⚕️ Disclaimer

Educational use only. This tool is **not a medical device** and must not be used for diagnosis or treatment decisions.

## 📄 License

Educational use only.
