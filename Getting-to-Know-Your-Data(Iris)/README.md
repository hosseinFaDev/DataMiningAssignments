# Data Mining - Homework 1

## Overview
This repository contains the solution for **Data Mining Assignment 1: Getting to Know Your Data**. The assignment involves data exploration, statistical analysis, correlation analysis, and data visualization using the **Iris dataset**.

## Features
- **Univariate Summary Statistics**: Calculation of key statistical measures for each species in the dataset.
- **Correlation Analysis**: Computation of Pearson’s correlation matrix for the dataset attributes.
- **Visualization**: Various plots including histograms, box plots, scatter plots, and probability distribution functions to analyze feature relationships.

## Dataset
The **Iris dataset** is used in this assignment. It contains measurements of 150 iris flowers belonging to three species:
- *Iris Setosa*
- *Iris Versicolor*
- *Iris Virginica*

Each instance includes:
- Sepal length
- Sepal width
- Petal length
- Petal width
- Species label

## Project Structure
```
Getting-to-Know-Your-Data(Iris)/
│── src/
│   ├── run.py
│── dist/
│   ├── correlations.csv
│   ├── statistics.csv
│── data/
│── requirements.txt(need to download from : https://archive.ics.uci.edu/dataset/53/iris)
```
- **`src/`**: Contains Python scripts for data processing, analysis, and visualization.
- **`dist/`**: Stores the output files (correlation matrix and statistical results).
- **`data/`**: Contains the dataset (if applicable).
- **`requirements.txt`**: Lists required Python libraries.

## Dataset is a public dataset and available for download in:
https://archive.ics.uci.edu/dataset/53/iris

## Usage
Run the main script to execute all analyses:
```bash
python src/run.py
```
- This script will:
  - Load the dataset
  - Compute statistical metrics
  - Generate correlation matrix
  - Produce various visualizations
  - Save results in `dist/` directory

## Acknowledgment
This project was completed with the assistance of **ChatGPT** for code generation and debugging support.

## License
This repository is for academic purposes only. Unauthorized use or distribution is not permitted.



