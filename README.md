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

## 🚀 Usage

### Running the Notebook

jupyter notebook res.ipynb

- Place your dataset in Training/ and Testing/ folders.  
- Run all cells to train, evaluate, and visualize results.

### Using the Pretrained Model

- Download the pretrained model via the Google Drive link above.  
- Load it directly in the notebook to evaluate without retraining.
