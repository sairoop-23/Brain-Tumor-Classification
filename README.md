# ​ Brain Tumor Classification

A deep learning project for classifying brain MRI images into four categories: **glioma, meningioma, pituitary tumor, and no tumor**. Includes a trained model and a **FastAPI backend** for serving predictions.

---

## ​ Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Model](#model) 
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [FastAPI Integration](#fastapi-integration)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)

---

## ​ Overview

This repository uses **transfer learning** (ResNet-50 and other CNNs) to classify brain tumor MRI scans. It covers:

- Data preprocessing  
- Model training and evaluation  
- Visualization of metrics  
- API deployment with FastAPI  

---

## ​ Dataset

- **Source:** Brain Tumor MRI Dataset (Kaggle)  
- **Classes:**
  - Glioma  
  - Meningioma  
  - Pituitary Tumor  
  - No Tumor  

---

## ​ Model

- Uses **ResNet-50 (pre-trained on ImageNet)** and fine-tuned CNNs.  
- Pretrained model download: *(https://colab.research.google.com/drive/1AMkfaPOxKA6IPwJFPEDbd2zFYsNuTPzR)*

---

## ​​ Installation

```bash
git clone https://github.com/sairoop-23/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification

python -m venv venv
# Activate virtual environment
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```


## Usage
### Running the Notebook
jupyter notebook res.ipynb

- Place your dataset in Training/ and Testing/ folders.  
- Run all cells to train, evaluate, and visualize results.

### Using the Pretrained Model

- Download the pretrained model via the Google Drive link above.  
- Load it directly in the notebook to evaluate without retraining.


## Project Structure

Brain-Tumor-Classification/
│
├── api/                  # FastAPI integration
│   ├── main.py           # FastAPI app
│   └── requirements.txt  # API dependencies
│
├── results/              # Confusion matrix, plots, metrics
├── res.ipynb             # Main Jupyter Notebook
├── requirements.txt      # Dependencies
├── dataset preview.png   # Dataset sample preview
└── README.md             # Documentation

## FastAPI Integration

### Run API

cd api
pip install -r requirements.txt
uvicorn main:app --reload

- Swagger UI → http://127.0.0.1:8000/docs
- ReDoc → http://127.0.0.1:8000/redoc

### Endpoints

Root
GET /

Predict
POST /predict

Example Request:

curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_image.jpg"

Example Response:

{
  "class": "Pituitary",
  "confidence": 0.9742
}

## Results

- Accuracy: ~98% (ResNet-50)
- Metrics: Accuracy, Precision, Recall
- Outputs: confusion matrix, sample predictions, plots (see results/ folder)
