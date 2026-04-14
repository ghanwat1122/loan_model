# 🏦 Bank Loan Prediction — Classification Model

A machine learning project that predicts whether a bank loan application will be approved or rejected, using a Decision Tree Classifier with pruning optimization.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow](#workflow)
- [Model Details](#model-details)
- [Results](#results)
- [Model Export](#model-export)
- [Technologies Used](#technologies-used)

---

## Overview

This project builds an end-to-end binary classification pipeline to predict loan approval status (`Yes` / `No`) based on applicant data. It covers the full ML lifecycle — from data preprocessing and EDA through model building, pruning, evaluation, and deployment export.

---

## Dataset

- **File:** `Bank_Loan.csv`
- **Target Variable:** `Loan_Status` (Yes = Approved, No = Rejected)
- **Dropped Column:** `Loan_ID` (irrelevant identifier)
- **Feature Types:** Mix of numerical and categorical variables

> ⚠️ Update the dataset path in the notebook before running locally.

---

## Project Structure

```
├── 1_Bank_Loan_Prediction_-_Deployment.ipynb   # Main notebook
├── Bank_Loan.csv                                # Dataset (add your own)
├── build.pkl                                    # Exported trained model
└── README.md
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/bank-loan-prediction.git
   cd bank-loan-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn plotly
   ```

3. **Add your dataset** — Place `Bank_Loan.csv` in the project root and update the file path in Cell 2 of the notebook.

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

## Workflow

The notebook follows a structured, step-by-step pipeline:

| Step | Description |
|------|-------------|
| 1 | Import libraries and load dataset |
| 2 | Data preprocessing — drop irrelevant columns, handle missing values, encode categoricals |
| 3 | Exploratory Data Analysis (EDA) — outlier detection, churn rate pie chart |
| 4 | Data partitioning — 70% train / 30% test split |
| 5 | Model building — Decision Tree (Gini index) |
| 6 | Tree visualization |
| 7 | Train set predictions & performance evaluation |
| 8 | **Model improvement via pruning** (min samples, max depth) |
| 9 | Test set predictions & final evaluation |
| 10 | Export model with `pickle` |

---

## Model Details

### Base Model
```python
DecisionTreeClassifier()   # Default: Gini impurity, unpruned
```

### Pruned Model (Final)
```python
DecisionTreeClassifier(
    criterion='gini',
    min_samples_split=300,   # Minimum samples required to split a node
    min_samples_leaf=50,     # Minimum samples required at a leaf node
    max_depth=4              # Maximum tree depth
)
```

Pruning was applied to reduce overfitting and improve generalization on unseen data.

---

## Results

Model performance was evaluated using a **Classification Report** (Precision, Recall, F1-Score) on both training and test datasets.

| Dataset | Model |
|---------|-------|
| Train | Pruned Decision Tree |
| Test | Pruned Decision Tree |

> See the notebook output cells for the full classification report metrics.

---

## Model Export

The final trained model is saved using `pickle` for deployment:

```python
import pickle
pickle.dump(dt2, open('build.pkl', 'wb'))
```

To load and use the model for inference:

```python
import pickle
model = pickle.load(open('build.pkl', 'rb'))
prediction = model.predict(new_data)
```

---

## Technologies Used

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-lightblue?logo=plotly)

- **Python 3.x**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **scikit-learn** — ML model, preprocessing, evaluation
- **matplotlib / seaborn** — static visualizations
- **plotly** — interactive EDA charts
- **pickle** — model serialization

---

## 📌 Notes

- Categorical variables were encoded using `LabelEncoder` before model training.
- The dataset contains a class imbalance (visible in the Loan Status pie chart) — consider techniques like SMOTE or class weighting for further improvement.
- This project is intended as a learning/deployment reference for end-to-end ML classification pipelines.