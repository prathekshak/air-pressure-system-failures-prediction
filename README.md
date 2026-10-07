# 🚛 APS Failure Prediction — Scania Trucks Dataset

This project aims to predict **failures in the Air Pressure System (APS)** of Scania trucks using machine learning models.  
The dataset was provided by **Scania CV AB** as part of the **Industrial Challenge 2016** (IDA 2016).

---

## 📦 Dataset Overview

**Dataset Name:** APS Failure and Operational Data for Scania Trucks  
**Source:** Scania CV AB, Sweden  
**License:** GNU General Public License v3.0  
**Link:** [https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks+Data+Set](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks+Data+Set)

### 📊 Key Information
- **Instances:** 60,000 (train) + 16,000 (test)
- **Attributes:** 171 anonymized operational sensor readings
- **Positive class (1):** Trucks with APS component failure (~1.7%)
- **Negative class (0):** Trucks with other failures
- **Missing values:** Represented as `"na"`

---

## ⚙️ Problem Description

The task is a **binary classification**:
- **1:** APS component failure (positive)
- **0:** Other failure (negative)

Because the dataset is **highly imbalanced**, model evaluation must consider the **cost of misclassification**:

| Predicted \ True | Positive | Negative |
|------------------|-----------|-----------|
| **Positive** | — | Cost₁ = 10 |
| **Negative** | Cost₂ = 500 | — |

💰 **Total Cost = (10 × False Positives) + (500 × False Negatives)**

---

## 🧹 Data Preprocessing

1. **Loaded dataset** and replaced `"na"` with `NaN`.  
2. **Imputed missing values** using median values per feature.  
3. **Scaled data** using `StandardScaler` for model consistency.  
4. **Train-test split:** 80–20 ratio with stratified sampling.  
5. **Handled imbalance** using `class_weight='balanced'` where applicable.

---

## 🧠 Models Used

| Model | Library | Notes |
|--------|----------|--------|
| **Logistic Regression** | `sklearn.linear_model` | Interpretable baseline |
| **Random Forest** | `sklearn.ensemble` | Robust ensemble model |
| **Gradient Boosting** | `sklearn.ensemble` | Sequential boosting of weak learners |
| **XGBoost** | `xgboost` | Optimized gradient boosting algorithm |

---

## 🚀 Training & Evaluation

Each model was trained on the preprocessed dataset and evaluated on the test set using:
- **Accuracy**
- **Precision / Recall / F1-score**
- **Confusion Matrix**
- **Total Cost Metric**

---

## 📈 Model Performance Comparison

| Model | Accuracy | Precision (Pos) | Recall (Pos) | F1-Score (Pos) | FP | FN | **Total Cost** |
|--------|-----------|----------------|---------------|----------------|----|----|----------------|
| **Logistic Regression** | 0.974 | 0.38 | 0.81 | 0.51 | 943 | 133 | **₹99,430** |
| **Random Forest** | 0.989 | 0.81 | 0.46 | 0.59 | 74 | 376 | **₹193,740** |
| **XGBoost** | 0.990 | 0.95 | 0.43 | 0.59 | 15 | 399 | **₹201,490** |

🧮 **Cost Formula:**  
`Total Cost = (10 × False Positives) + (500 × False Negatives)`

---

### 🔍 Observations
- **Logistic Regression** achieves the **lowest total cost**, even though its accuracy is slightly lower.  
- **XGBoost** and **Random Forest** show superior raw accuracy but incur **higher penalties** due to missed failures.  
- The **positive class (failures)** is small, so recall is critical — missing a failure costs 50× more than a false alarm.

---

### 📊 Confusion Matrices

| Model | Confusion Matrix |
|--------|------------------|
| **Logistic Regression** | `[[40357, 943], [133, 567]]` |
| **Random Forest** | `[[41226, 74], [376, 324]]` |
| **XGBoost** | `[[41285, 15], [399, 301]]` |

---

## 🧮 Cost Analysis Visualization (Optional Code)

```python
import matplotlib.pyplot as plt

models = ['Logistic Regression', 'Random Forest', 'XGBoost']
costs = [99430, 193740, 201490]

plt.bar(models, costs)
plt.ylabel('Total Cost (Lower is Better)')
plt.title('Cost-based Comparison of Models')
plt.show()
