# SmartCart Customer Segmentation

An end-to-end unsupervised machine learning project that segments retail customers into distinct groups based on demographics, spending behavior, and engagement.

## Overview

Businesses need fast, data-driven ways to understand who their customers are so they can target marketing, offers, and retention differently across groups. This project uses clustering to discover natural customer segments directly from customer data — with no labels — using applicant income, age, tenure, spending across product categories, household size, and campaign engagement.

## Dataset

- Retail customer records (`smartcart_customers.csv`)
- Features include: Income, Year of Birth, Education, Marital Status, Kidhome/Teenhome, Date Joined, Recency, Response, and spend across Wines, Fruits, Meat, Fish, Sweets, and Gold products
- Engineered features: `age`, `cust_tenure_day`, `tot_spend`, `tot_child`

## Features

- Data cleaning and preprocessing (handling missing income values, outlier removal on age/income)
- Feature engineering (age, tenure, total spend, total children at home)
- Category simplification (Education, Marital Status grouped into fewer meaningful classes)
- Exploratory Data Analysis (EDA) with pairplots and a correlation heatmap
- One-hot encoding and feature scaling with StandardScaler
- Dimensionality reduction with PCA
- Cluster count selection via the elbow method (KneeLocator)
- Clustering with K-Means and Agglomerative (Ward linkage), compared side by side
- Cluster profiling by income, spend, age, and family size

## Tech Stack

- Python
- Pandas
- Matplotlib, Seaborn
- Scikit-learn
- kneed
- Jupyter Notebook

## Project Workflow

1. Data Collection
2. Data Cleaning & Missing Value Handling
3. Feature Engineering
4. Category Simplification
5. Outlier Removal
6. Exploratory Data Analysis
7. Encoding & Feature Scaling
8. Dimensionality Reduction (PCA)
9. Optimal Cluster Count Selection
10. Clustering (K-Means vs. Agglomerative)
11. Cluster Profiling & Interpretation

## Results

Both K-Means and Agglomerative Clustering (Ward linkage) were applied to the PCA-reduced feature space, with the optimal cluster count identified via the elbow method:

| Method | Clusters | Notes |
|---|---|---|
| **K-Means** | 4 | Elbow point confirmed via KneeLocator |
| **Agglomerative (Ward)** | 4 | Used for final cluster profiling |

**Takeaway:** The clustering revealed 4 distinct customer segments with clearly different income, spending, and family-size profiles — providing a basis for targeted marketing strategies (e.g. distinguishing high-value, low-engagement customers from budget-conscious families).

## How to Run

1. Clone the repository
```
git clone https://github.com/K-BHAGAT/SmartCart.git
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Launch Jupyter Notebook
```
jupyter notebook
```

4. Open `smartcart.ipynb` and run all cells

## Future Improvements

- Validate cluster count with silhouette score in addition to the elbow method
- Add descriptive cluster labels (e.g. "High-value, no kids") for easier business use
- Explore alternative clustering methods (DBSCAN, Gaussian Mixture Models)
- Deploy an interactive dashboard (Streamlit) for exploring segments

## Visualizations

![Correlation Heatmap](heatmap.png)

## About

An end-to-end machine learning project for customer segmentation using clustering techniques to identify customer groups and enable data-driven marketing strategies.
