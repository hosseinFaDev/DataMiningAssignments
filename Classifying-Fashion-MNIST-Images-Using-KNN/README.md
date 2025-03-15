# Data Mining - Homework 3

## Overview
This repository contains the solution for **Data Mining Assignment 3: Classifying Fashion MNIST Images Using KNN**. The assignment involves implementing the **K-Nearest Neighbors (KNN) algorithm from scratch** and applying it to the **Fashion MNIST dataset**.

## Features
- **Data Exploration**: Analyzing image data structure and label distribution.
- **Data Preprocessing**:
  - Min-Max normalization of pixel values
  - Principal Component Analysis (PCA) for dimensionality reduction
- **KNN Implementation**:
  - Custom implementation from scratch (without scikit-learn’s KNN)
  - Distance-weighted voting mechanism
  - Handling ties in classification
- **Hyperparameter Tuning**:
  - Experimenting with different `k` values and PCA variance ratios
  - Identifying optimal settings for classification accuracy
- **Model Evaluation**:
  - Computing precision, recall, F1-score, and AUC per class
  - Visualizing confusion matrix and ROC curves

## Dataset
The **Fashion MNIST dataset** is a publicly available benchmark dataset:
- **Source**: [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- **Description**: 70,000 grayscale images (28x28 pixels) of clothing items, classified into 10 categories.
- **Classes**: T-shirt/top, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot.
- **Note**: To ensure compliance with licensing, do **not** upload the dataset to a public repository. Instead, fetch it using `fetch_openml()` from `sklearn.datasets`.

## Project Structure
```
Classifying-Fashion-MNIST-Images-Using-KNN/
│── code.ipynb
│── requirements.txt
```
- **`code.ipynb`**: Jupyter Notebook containing all implementation steps.
- **`requirements.txt`**: List of required Python libraries.
- **Report PDF**: Summarizes methodology, results, and visualizations.


## Usage
Run the Jupyter Notebook to execute all steps:
```bash
jupyter notebook code.ipynb
```

## Acknowledgment
This project was completed with the assistance of **ChatGPT** for debugging and optimization.

## License & Legal Considerations
- The **Fashion MNIST dataset** is publicly available and should be downloaded dynamically.
- This repository is for academic purposes only. Unauthorized distribution is not permitted.

