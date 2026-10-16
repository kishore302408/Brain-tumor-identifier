# NeuroScan Pro — Brain Tumor MRI Classification

MED-02 Medical AI Challenge: Brain Tumor Classification / Medical Image Analysis from MRI Scans

## Overview
Web-based AI tool for classifying brain MRI scans into glioma, meningioma, notumor, pituitary with Grad-CAM visualisations, confidence scores, ensemble predictions and clinical metrics.

## Features
* Upload single or multiple MRI images
* Three models: CNN, ResNet18, MobileNetV2
* Grad-CAM heatmaps
* Ensemble prediction
* Parametric features view
* Segmentation overlay toggle (demo)
* Performance metrics: accuracy, confusion matrix, classification report
* Comparison table + CSV report export
* Professional Streamlit UI + Flask API

## Quick Start
```bash
pip install -r requirements.txt
python train.py path/to/dataset 8
python -m streamlit run streamlit_app.py --server.port 8502
# or
python app.py  # Flask on port 5001
```

## Dataset
Training/Testing with folders: glioma, meningioma, notumor, pituitary

## Results
* CNN: 80.13%
* ResNet18: 95.13%
* MobileNetV2: 95.00%

## License
Educational use only, not a diagnostic tool.
