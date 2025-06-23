# Predicting Non-B DNA Structure Using PacBio Interpulse Durations

> First study to predict non-canonical DNA structures using PacBio Interpulse Durations and machine learning methods.

## Overview

This project implements a machine learning pipeline to detect **non-B DNA structures** using **interpulse duration (IPD)** data from **Pacific Biosciences (PacBio)** sequencing. Non-B DNA structures—such as G-quadruplexes, Z-DNA, and inverted repeats—are associated with mutagenesis, oncogenesis, and genome instability.

We apply time-series clustering, anomaly detection, and supervised classification on IPD data to distinguish B-DNA from non-B DNA with high precision.

## Project Pipeline

### Data Source
- **PacBio sequencing data** of B-DNA and non-B DNA (various subtypes).
- Forward and reverse strand **interpulse duration (IPD)** values.
- B-DNA: 56,000+ samples  
- Non-B DNA: 330,000+ samples (downsampled & truncated for consistency)

### Preprocessing
- Parse `.pkl` files with IPD time series per read.
- Truncate sequences to 100 base pairs.
- Balance dimensionality across DNA types (200D vectors from forward + reverse).

### Clustering (Unsupervised Learning)
- **K-Means** and **K-Shape** clustering to explore latent structure.
- Use **Silhouette Score** to find optimal `k` (usually `k=2`).
- Project high-dimensional IPD vectors via **UMAP** for visualization.

### Anomaly Detection
- Train **One-Class SVM** on B-DNA IPDs to detect structural deviations.
- Grid search over `gamma` hyperparameter (0.001, 100).
- Assign labels (`0` for normal/B-DNA, `1` for anomalous/non-B) to samples.

### Classification
- Use **Logistic Regression** to classify B DNA vs. non-B DNA using SVM-labeled data.
- Evaluate using **accuracy, precision, recall, F1-score**.
- Achieved final **accuracy: 98.57%**.

### Evaluation Graphs
- Compare classifier performance across gamma values.
- Visualize metrics (precision, recall, F1-score) for anomalies vs normal samples.

## 🔍 Key Results

| Metric           | B-DNA (0) | Non-B DNA (1) |
|------------------|-----------|----------------|
| **Precision**     | 0.99      | 0.96           |
| **Recall**        | 1.00      | 0.89           |
| **F1 Score**      | 0.99      | 0.93           |
| **Accuracy**      | 0.9857    | —              |

- Optimal `gamma` for One-Class SVM: **0.001**
- KMeans Silhouette Score: ~0.31  
- KShape Silhouette Score: ~0.47


## References

This work builds on prior research on non-canonical DNA structures, polymerase kinetics, and genome instability.

> Key references:  
> - Guiblet et al., 2021  
> - Sawaya et al., 2015  
> - Hosseini et al., 2023  
> - Makova et al., 2023

## Research Team

- **Naavya Jain** – Research project, model development, and analysis  
- **Dr. Derek Aguiar** – Faculty mentor, University of Connecticut  
- **Marjan Hosseini** – PhD mentor, data preprocessing

## Acknowledgments

This project was supported by the University of Connecticut Advanced Research Mentorship Program. Special thanks to Dr. Aguiar and Marjan Hosseini for their mentorship.
