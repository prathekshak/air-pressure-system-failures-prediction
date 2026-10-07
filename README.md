# 🚛 Scania APS Failure Prediction

### Cost-Sensitive Machine Learning for Predictive Maintenance

This project applies machine learning to predict **Air Pressure System (APS) component failures** in Scania trucks.

The project uses the **APS Failure and Operational Data for Scania Trucks** dataset provided by **Scania CV AB** as part of the **Industrial Challenge 2016 (IDA 2016)**. The primary focus is not only classification accuracy, but also **cost-sensitive evaluation**, where missing an actual APS failure is significantly more expensive than generating a false alarm.

> **Key Finding:** Logistic Regression achieved the **lowest total misclassification cost**, despite having lower overall accuracy than Random Forest and XGBoost.

---

## 📦 Dataset

**Dataset:** APS Failure and Operational Data for Scania Trucks  
**Source:** Scania CV AB, Sweden  
**License:** GNU General Public License v3.0

The dataset contains operational sensor measurements collected from Scania trucks and is used to predict whether a failure is associated with the vehicle's **Air Pressure System (APS)**.

### Dataset Overview

| Property | Description |
|---|---|
| Training instances | 60,000 |
| Test instances | 16,000 |
| Features | 171 anonymized operational attributes |
| Positive class | APS component failure |
| Negative class | Other failure |
| Missing values | Represented as `"na"` |
| Positive class proportion | ~1.7% |

---

# 🎯 Problem Statement

The task is a **binary classification problem**:

```text
1 → APS component failure
0 → Other failure
```

The dataset is highly imbalanced, making conventional accuracy an insufficient evaluation metric.

In a predictive-maintenance scenario, **missing an actual APS failure can be substantially more costly than incorrectly flagging a healthy vehicle**.

---

## 💰 Cost-Sensitive Evaluation

The project uses the following misclassification costs:

| Prediction | Actual Positive | Actual Negative |
|---|---:|---:|
| **Positive** | — | Cost₁ = 10 |
| **Negative** | Cost₂ = 500 | — |

Therefore:

```text
Total Cost =
    (10 × False Positives)
    + (500 × False Negatives)
```

A false negative costs **50× more** than a false positive.

This makes **failure detection and recall** particularly important.

---

# 🧹 Data Preprocessing

The following preprocessing pipeline was applied:

### 1. Missing Value Handling

The dataset represents missing values using:

```text
"na"
```

These values were converted to `NaN` and imputed using the **median value of each feature**.

### 2. Feature Scaling

Features were standardized using:

```python
StandardScaler
```

### 3. Train-Test Split

The data was divided using an **80/20 stratified split** to preserve the class distribution.

### 4. Class Imbalance

Class imbalance was addressed using:

```python
class_weight="balanced"
```

where supported by the selected model.

---

# 🧠 Machine Learning Models

Four classification approaches were considered:

| Model | Library | Purpose |
|---|---|---|
| **Logistic Regression** | `sklearn.linear_model` | Interpretable baseline |
| **Random Forest** | `sklearn.ensemble` | Robust ensemble model |
| **Gradient Boosting** | `sklearn.ensemble` | Sequential boosting |
| **XGBoost** | `xgboost` | Optimized gradient boosting |

The reported performance comparison focuses on **Logistic Regression, Random Forest, and XGBoost**.

---

# 📊 Evaluation Metrics

Each model was evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **Total Misclassification Cost**

The most important metric for this problem is:

> **Total Cost**, because false negatives carry a significantly higher penalty than false positives.

---

# 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | FP | FN | **Total Cost** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 0.974 | 0.38 | **0.81** | 0.51 | 943 | 133 | **₹99,430** |
| Random Forest | **0.989** | 0.81 | 0.46 | 0.59 | 74 | 376 | ₹193,740 |
| XGBoost | **0.990** | **0.95** | 0.43 | 0.59 | 15 | 399 | ₹201,490 |

### Cost Formula

```text
Total Cost =
    (10 × False Positives)
    + (500 × False Negatives)
```

---

# 🔍 Key Findings

### 🥇 Logistic Regression — Lowest Cost

Logistic Regression achieved the **lowest total cost of ₹99,430**.

Although its accuracy was lower than the ensemble models, it achieved substantially higher recall:

```text
Recall = 0.81
False Negatives = 133
```

This is important because every missed APS failure carries a cost of **500**.

### 🌲 Random Forest

Random Forest achieved:

```text
Accuracy = 98.9%
Recall = 46%
False Negatives = 376
Total Cost = ₹193,740
```

Its high accuracy does not translate into the lowest operational cost because of the larger number of missed failures.

### ⚡ XGBoost

XGBoost achieved the highest reported accuracy:

```text
Accuracy = 99.0%
Precision = 95%
```

However, it produced:

```text
False Negatives = 399
Total Cost = ₹201,490
```

This demonstrates why **accuracy and precision alone are not sufficient** for highly imbalanced, cost-sensitive classification problems.

---

# 🧮 Confusion Matrices

| Model | Confusion Matrix |
|---|---|
| **Logistic Regression** | `[[40357, 943], [133, 567]]` |
| **Random Forest** | `[[41226, 74], [376, 324]]` |
| **XGBoost** | `[[41285, 15], [399, 301]]` |

For each matrix:

```text
[[True Negatives, False Positives],
 [False Negatives, True Positives]]
```

---

# 📊 Cost Comparison

The cost difference between the models highlights the importance of choosing an evaluation metric aligned with the real-world problem.

```text
Logistic Regression  ████████████████████  ₹99,430
Random Forest        █████████████████████████████████████  ₹193,740
XGBoost              ████████████████████████████████████████ ₹201,490
```

### Python Visualization

```python
import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

costs = [
    99430,
    193740,
    201490
]

plt.bar(models, costs)
plt.ylabel("Total Cost (Lower is Better)")
plt.title("Cost-Based Model Comparison")
plt.show()
```

---

# 🔄 Machine Learning Workflow

```text
                    Raw Dataset
                         │
                         ▼
              Missing Value Handling
                         │
                         ▼
                 Median Imputation
                         │
                         ▼
                  Feature Scaling
                         │
                         ▼
              Stratified Train/Test Split
                         │
                         ▼
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
       Logistic Regression    Random Forest
                │                 │
                └────────┬────────┘
                         │
                         ▼
                     XGBoost
                         │
                         ▼
              Model Evaluation
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
     Recall           F1-Score       Total Cost
                         │
                         ▼
                Best Cost Model
```

---

# 🛠️ Technology Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- XGBoost

### Data Processing

- NumPy
- Pandas
- StandardScaler
- Median imputation

### Visualization

- Matplotlib

### Models

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

---

# 📁 Suggested Project Structure

```text
scania-aps-failure-prediction/
│
├── data/
│   ├── aps_failure_training_set.csv
│   └── aps_failure_test_set.csv
│
├── notebooks/
│   └── aps_failure_prediction.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
│
├── results/
│   ├── confusion_matrices/
│   └── model_comparison.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

Install Python 3.9+ and the required dependencies.

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

## Run the Project

Clone the repository:

```bash
git clone https://github.com/<your-username>/scania-aps-failure-prediction.git
cd scania-aps-failure-prediction
```

Run the notebook or training pipeline:

```bash
jupyter notebook
```

---

# 📚 Dataset Source

The project uses the **APS Failure at Scania Trucks** dataset provided by Scania CV AB.

Dataset information:

- **60,000 training instances**
- **16,000 test instances**
- **171 anonymized features**
- Highly imbalanced failure class
- Missing values represented as `"na"`

---

# 💡 Key Takeaways

This project demonstrates several important machine learning concepts:

- Handling **high-dimensional tabular data**
- Missing-value imputation
- Feature standardization
- Class imbalance
- Binary classification
- Ensemble learning
- Cost-sensitive evaluation
- Precision vs. recall trade-offs
- Confusion matrix analysis
- Selecting models based on **business impact rather than accuracy alone**

### Most Important Insight

> **The model with the highest accuracy is not necessarily the best model.**

For predictive maintenance, failing to identify a real APS failure is significantly more costly than raising a false alarm. As a result, **Logistic Regression achieved the best overall cost performance despite having lower accuracy than Random Forest and XGBoost.**

---

## 👩‍💻 Author

**Pratheksha Kanagaraj**

Computer Science Graduate Student · Software Engineer · AI/ML Enthusiast

---

⭐ **If you found this project useful, consider giving the repository a star!**
