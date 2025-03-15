## Assignment 4 - Data Mining

### Instructor: Dr. Mostafa Fakhrahmad

### Overview
This project involves generating synthetic datasets and implementing the Affinity Propagation (AP) clustering algorithm. The performance of AP is compared with other clustering methods like K-means and HDBSCAN.

### Datasets
The synthetic datasets are generated using `scikit-learn` functions:
- `make_circles`: Creates concentric circle clusters.
- `make_moons`: Generates two crescent-shaped clusters.
- `make_blobs`: Produces Gaussian-distributed clusters with predefined centers.

These datasets are artificially generated and do not contain any proprietary or sensitive data. Therefore, sharing them publicly does not pose any legal concerns.

### Implementation
1. **Data Synthesis**: Three datasets are generated with predefined parameters.
2. **Data Exploration**: Scatter plots are used to visualize the datasets.
3. **Clustering Algorithm**: AP is implemented from scratch without using pre-built library functions.
4. **Hyperparameter Selection**: Different hyperparameters are tested to optimize clustering performance.
5. **Evaluation**: Clustering quality is assessed using multiple evaluation metrics.
6. **Comparison**: Results of AP clustering are compared with K-means and HDBSCAN.

### Installation
To run this project, install the required dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn hdbscan
```

### Usage
Run the Jupyter Notebook (`code.ipynb`) to generate datasets, perform clustering, and visualize results.

### Disclaimer
This project was completed as part of an academic assignment. If you intend to use or modify this project, ensure compliance with your institution’s academic policies. The synthetic datasets used in this project are publicly generated and free to share.

### License
This project is for educational purposes only and does not include any proprietary or confidential content.

