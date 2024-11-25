# Customer Churn Prediction

This project predicts whether a bank customer will churn (leave the bank) using various machine learning models.

## Folders
- `data/`: Contains the dataset.
- `models/`: Stores trained machine learning models.
- `notebooks/`: Jupyter notebooks for EDA and training.
- `results/`: Stores evaluation metrics and plots.
- `scripts/`: Python scripts for preprocessing and training.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `app.py` to start the API.

## API
- POST `/predict`: Predicts churn using customer features.

# Customer Churn Prediction API

## Project Overview

This project is a **machine learning-based API** that predicts customer churn (whether a customer will leave or stay) using a trained **Random Forest model**. The API is built with **Flask** and provides endpoints for making single or batch predictions.

### Features
- Pre-trained machine learning model (Random Forest) for churn prediction.
- RESTful API using Flask.
- Input validation and support for batch predictions.
- Deployment-ready structure.

---

## Installation and Setup

Follow these steps to set up and run the project locally:

### Prerequisites
- Python 3.11 or higher.
- Required Python libraries (listed in `requirements.txt`).

### Steps
1. Clone the repository:
   ```bash
   git clone <CodSoft_Repo_URL>
   cd Customer-Churn-Prediction

