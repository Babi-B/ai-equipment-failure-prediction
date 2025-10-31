# AI Equipment Failure Prediction

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.30-green)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

**Repository:** [https://github.com/Babi-B/ai-equipment-failure-prediction](https://github.com/Babi-B/ai-equipment-failure-prediction)
---
## Overview
This project implements an **AI-based predictive maintenance system** for industrial equipment.  
It leverages historical sensor data and operational logs to predict equipment failures, helping reduce downtime and maintenance costs.

Key components include:

- **Exploratory Data Analysis (EDA)**: Analysis of sensor and operational datasets.
- **Preprocessing Utilities**: Scripts to clean and transform raw data for modeling.
- **Trained Models**: Machine learning models trained on historical datasets for failure prediction.
- **Streamlit UI**: Interactive web interface to visualize predictions and model performance.
- **Test Metrics**: Scripts and CSV results to evaluate model performance.
---
## Features
- Predict failures using historical sensor data
- Visualize operational conditions and failure trends
- Interactive Streamlit interface for real-time predictions
- Easy integration of new data and retraining of models
- Handles multiple datasets (FD001, etc.)
---
```text
├── .gitignore
├── .gitattributes
├── README.md
├── requirements.txt
├── preprocessing/ # Data preprocessing scripts
├── models/ # Saved trained models
├── data/ # Raw and processed datasets
├── notebooks/ # EDA and model exploration
├── streamlit_app/ # Streamlit interface
└── test_metrics/ # Evaluation scripts and CSV results
```
---
## Installation
1. **Clone the repository**
```bash
git clone https://github.com/Babi-B/ai-equipment-failure-prediction.git
cd ai-equipment-failure-prediction
```
2. **Create a virtual environment**
```bash
python -m venv .venv
```
3. **Activate the virtual environment**
- Windows (Git Bash):
```bash
source .venv/Scripts/activate
```
- Windows (PowerShell):
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```
4. **Install dependencies**
```bash
pip install -r requirements.txt
```
---
## Usage
**Run the Streamlit UI**
```bash
streamlit run streamlit_app/main.py
```
Open your browser at the link provided by Streamlit (usually `http://localhost:8501`).

**Run preprocessing scripts**
```bash
python preprocessing/preprocess.py
```
**Train models**
```bash
python models/train_model.py
```
**Evaluate models**
```bash
python test_metrics/evaluate.py
```
---
**Git Large File Storage (LFS)**
Large model files and datasets are tracked using Git LFS
If you clone the repo, make sure you have Git LFS installed:
```bash
git lfs install
git lfs pull
```


