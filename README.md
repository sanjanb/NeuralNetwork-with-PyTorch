# Churn Prediction Model with PyTorch Lightning

## Project Overview

This project focuses on developing a churn prediction model using PyTorch Lightning. The goal is to predict whether bank customers will churn (exit) based on various features such as credit score, age, and tenure. The project leverages PyTorch Lightning to simplify the training process and improve model management.

## Objectives

- Set up a machine learning environment with necessary libraries.
- Load and explore the churn modeling dataset.
- Preprocess data using encoding and scaling techniques.
- Build and train a neural network using PyTorch Lightning.
- Evaluate model performance using metrics and visualizations.

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Development](#model-development)
5. [Training and Evaluation](#training-and-evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Virtual environment setup

### Installation

1. **Create a Virtual Environment:**
   ```bash
   python -m venv ml_env
   ```

2. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     ml_env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source ml_env/bin/activate
     ```

3. **Install Necessary Libraries:**
   ```bash
   pip install scikit-learn torch torchvision torchmetrics pandas seaborn matplotlib pytorch-lightning
   ```

## Dataset Overview

The churn modeling dataset contains information about bank customers, including features like credit score, age, tenure, and churn status. The dataset is used to train a neural network to predict customer churn.

## Data Preprocessing

### Steps

1. **Load Data:**
   - The dataset is loaded from a CSV file using pandas.

2. **Initial Data Inspection:**
   - Check the columns and data types to ensure they are appropriate for analysis.

3. **Data Cleaning:**
   - Drop irrelevant columns (e.g., row number, customer ID, last name).
   - Handle missing values using `dropna`.
   - Remove duplicates using `drop_duplicates`.

4. **Data Exploration:**
   - Check data types and visualize the distribution of the target variable (churn) to understand class imbalance.

## Model Development

### LightningDataModule

- **Purpose:** Manage and prepare data for training a neural network.
- **Key Methods:**
  - `__init__`: Initialize parameters like batch size.
  - `prepare_data`: Load and clean data.
  - `setup`: Split data into training and validation sets and apply transformations.
  - `train_dataloader` and `val_dataloader`: Create data loaders for training and validation.

### LightningModule

- **Purpose:** Encapsulate model-related operations.
- **Key Methods:**
  - `__init__`: Define network structure and hyperparameters.
  - `forward`: Define data flow through the network.
  - `training_step`: Define operations for a single batch of training data.
  - `validation_step`: Define validation process for a single batch of validation data.
  - `configure_optimizers`: Set up optimizers and learning rate schedulers.

## Training and Evaluation

### Training Process

- The model is trained using PyTorch Lightning's `Trainer` for a maximum of 20 epochs.
- Metrics such as validation accuracy and F1 score are logged during training.

### Evaluation Metrics

- **Accuracy:** Proportion of correct predictions.
- **Precision:** Proportion of positive identifications that are correct.
- **Recall:** Proportion of actual positives correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.

## Results

- The model's performance is evaluated using various metrics, and the results are visualized using Seaborn.
- The final evaluation metrics are computed to assess the model's performance on the validation data.

## Conclusion

This project demonstrates the use of PyTorch Lightning to simplify the training process of a neural network for churn prediction. By following the steps outlined in this report, you can effectively prepare and train a classification model on a churn modeling dataset.

---

**Note:** Ensure you have the `churn_data.csv` dataset available in your working directory to follow along with the code examples.
