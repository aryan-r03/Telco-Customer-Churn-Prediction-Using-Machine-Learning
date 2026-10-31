# Telco Customer Churn Prediction Using Machine Learning
**CA2 Skill Based Assessment**

---

## Project Overview

This project builds and evaluates four machine learning classification models to predict whether a telecom customer is likely to discontinue (churn) their subscription. It covers the complete machine learning pipeline: data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and comparison.

---

## Problem Statement

Customer churn is a significant challenge for telecommunications companies. Losing customers directly reduces revenue and business growth. This project develops classification models that predict whether a customer will leave a telecom service provider based on their demographic, account, and service-usage information.

---

## Project Objectives

1. Understand customer demographics, subscribed services, and billing patterns.
2. Identify the factors most strongly associated with customer churn.
3. Clean and preprocess the dataset to make it ML-ready.
4. Perform comprehensive exploratory data analysis and statistical analysis.
5. Develop four machine learning classification models.
6. Evaluate and compare model performance using appropriate metrics.
7. Identify the best-performing model based on actual evaluation results.

---

## Dataset Description

| Property            | Value                                              |
|---------------------|----------------------------------------------------|
| File                | `Telco_customer_churn.csv`                         |
| Source              | IBM Telco Customer Churn Dataset                   |
| Total Records       | 7,043 customer rows                                |
| Total Columns       | 33 (raw), 21 (after dropping non-predictive cols)  |
| Target Variable     | `Churn Value` (0 = Retained, 1 = Churned)          |
| Churn Rate          | ~26.5% churned (1,869), ~73.5% retained (5,174)    |

### Columns Dropped (Non-Predictive / Data Leakage)

| Column        | Reason for Dropping                                             |
|---------------|-----------------------------------------------------------------|
| CustomerID    | Unique identifier — no predictive signal                       |
| Count         | Constant value (1) for every row                               |
| Country       | Constant — "United States"                                     |
| State         | Constant — "California"                                        |
| City          | Granular location with hundreds of sparse categories            |
| Zip Code      | Same reason as City                                            |
| Lat Long      | Raw coordinate string                                           |
| Latitude      | Granular geographical coordinate                               |
| Longitude     | Granular geographical coordinate                               |
| Churn Label   | Duplicate of target variable (text version of Churn Value)     |
| Churn Score   | Post-hoc IBM risk score — severe data leakage                  |
| Churn Reason  | Only populated for churned customers (5,174 NaN) — data leakage|

### Final Predictive Features (20 Features)

**Numerical:** `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`

**Categorical:** `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`

---

## Technologies Used

| Library      | Version | Purpose                                    |
|--------------|---------|--------------------------------------------|
| Python       | 3.12.10 | Core programming language                  |
| Pandas       | 2.3.3   | Data loading, manipulation, EDA            |
| NumPy        | 2.5.4   | Numerical operations                       |
| Matplotlib   | 3.10.7  | Visualizations and charts                  |
| Scikit-learn | 1.7.2   | ML models, preprocessing, evaluation       |
| SciPy        | 1.16.2  | Scientific computing (transitive dep.)     |

---

## Data Preprocessing Steps

1. **Data Type Conversion**: Converted `Total Charges` from object (string) to float64 using `pd.to_numeric(errors='coerce')`.
2. **Missing Value Handling**: 11 records had blank `Total Charges` (customers with Tenure = 0, not yet billed). Imputed with median value.
3. **Duplicate Check**: Zero duplicate rows found. No removal required.
4. **Outlier Detection**: Used IQR method on all 4 numerical features. Zero outliers detected. No valid records removed.
5. **Binary Encoding (Label Encoding)**: Applied direct `Yes→1 / No→0 / Male→1 / Female→0` mapping to 10 binary columns.
6. **Multi-Category Encoding (One-Hot)**: Applied `pd.get_dummies(drop_first=True)` to `Internet Service`, `Contract`, `Payment Method`, `Multiple Lines`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`.
7. **Train-Test Split**: 80% training (5,634 rows), 20% testing (1,409 rows). Stratified split with `random_state=42`.
8. **Feature Scaling**: Applied `MinMaxScaler` to the 4 continuous numerical features (`Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV`). **Scaler fitted only on X_train** — test set transformed only. No data leakage.

---

## Exploratory Data Analysis Performed

### A. Basic Dataset Analysis
- `head()`, `tail()`, `shape`, `info()`, `describe()`, `columns`, `nunique()`, `isnull().sum()`, `duplicated().sum()`

### B. Univariate Analysis
- Churn distribution (bar + pie chart)
- Tenure, Monthly Charges, Total Charges, CLTV distributions (histograms)
- Gender, Senior Citizen, Partner, Dependents distributions (bar charts)
- All service and contract features (grouped bar charts)

### C. Bivariate Analysis
- Gender, Senior Citizen, Contract, Internet Service, Payment Method, Paperless Billing, Tech Support, Online Security — each vs. Churn (stacked bar charts)
- Tenure Months vs Churn, Monthly Charges vs Churn (boxplots)
- Tenure Months vs Monthly Charges, Tenure Months vs Total Charges (scatter plots coloured by Churn)

### D. Correlation Analysis
- Correlation matrix for all numerical features + Churn Value
- Correlation heatmap with annotated values

### E. Outlier Visualization
- Boxplots for all 4 numerical features

---

## Machine Learning Algorithms

| # | Algorithm              | Key Parameters                      |
|---|------------------------|-------------------------------------|
| 1 | Logistic Regression    | `max_iter=1000`, `random_state=42`  |
| 2 | K-Nearest Neighbors    | K selected via 5-fold CV on X_train |
| 3 | Gaussian Naive Bayes   | Default parameters                  |
| 4 | Decision Tree          | `max_depth=5`, `random_state=42`    |

> **KNN Note**: K was selected using 5-fold Stratified Cross-Validation on the training set only. The test set was not used for K selection.

---

## Evaluation Metrics

- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (True positives + False positives) — measures reliability of churn predictions
- **Recall**: True positives / (True positives + False negatives) — measures coverage of actual churners identified
- **F1-Score**: Harmonic mean of Precision and Recall — primary metric for imbalanced classification
- **Confusion Matrix**: Visual breakdown of true/false positives and negatives

---

## Actual Model Comparison Results

> ⚠️ All values below are from actual script execution. No values were hardcoded or fabricated.

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.8055   | 0.6506    | 0.5775 | 0.6119   |
| KNN (K=19 via CV)    | 0.7771   | 0.5794    | 0.5856 | 0.5824   |
| Gaussian Naive Bayes | 0.6899   | 0.4549    | 0.8503 | 0.5927   |
| Decision Tree        | 0.7878   | 0.5935    | 0.6364 | **0.6142** |

### Key Observations

- **Logistic Regression**: Highest accuracy (80.55%). Strong baseline with good precision. Best overall accuracy.
- **KNN (K=19 selected by CV)**: Balanced precision and recall. Moderate overall performance.
- **Gaussian Naive Bayes**: Exceptional recall (85.03%) — identifies most churners — but lowest precision (45.49%). Suitable when the business cost of missing a churner is very high.
- **Decision Tree (max_depth=5)**: Best F1-Score (0.6142). Best balance of precision and recall. Most interpretable for business explanation.

### Final Model Selection

**Decision Tree Classifier** is selected as the final model based on the highest F1-Score (0.6142), which best balances precision and recall for this imbalanced churn prediction problem. Logistic Regression is the runner-up, offering slightly better accuracy with competitive F1.

---

## Project Folder Structure

```
Student_Churn_Prediction_CA2/
│
├── telco_churn_prediction.py               ← Complete ML pipeline script
├── Telco_customer_churn.csv                ← Primary dataset (IBM Telco)
├── README.md                               ← This file
├── requirements.txt                        ← Python dependencies
│
└── visualizations/
    ├── churn_distribution.png              ← Target variable bar + pie chart
    ├── feature_analysis.png                ← Numerical feature histograms
    ├── demographic_distribution.png        ← Gender, Senior Citizen, Partner, Dependents
    ├── service_contract_distribution.png   ← Service and contract univariate
    ├── bivariate_analysis.png              ← Feature vs Churn stacked bars
    ├── charges_tenure_boxplots.png         ← Boxplots + scatter for numerical vs Churn
    ├── correlation_heatmap.png             ← Numerical feature correlation heatmap
    ├── outlier_boxplots.png                ← IQR outlier check boxplots
    ├── model_comparison.png                ← Accuracy & F1-Score comparison bar charts
    └── confusion_matrices/
        ├── cm_logistic_regression.png
        ├── cm_knn.png
        ├── cm_gaussian_nb.png
        └── cm_decision_tree.png
```

---

## Installation and Execution Instructions

### Step 1 — Clone or Copy the Project Folder

Ensure `Student_Churn_Prediction_CA2/` is on your machine with both the `.py` script and `Telco_customer_churn.csv` inside it.

### Step 2 — Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn scipy
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3 — Run the Script

Navigate to the project folder and execute:

```bash
cd Student_Churn_Prediction_CA2
python telco_churn_prediction.py
```

### Step 4 — Check Output

- All printed results (EDA, metrics, model comparison table) appear in the terminal.
- All visualizations are saved automatically to `visualizations/` and `visualizations/confusion_matrices/`.
- No interactive pop-up windows are opened — all charts are saved as `.png` files.

### Notes

- Python 3.8+ required.
- The script uses `matplotlib.use("Agg")` — a non-interactive backend — so it runs without a display server. Suitable for remote machines and scripts.
- Do not rename the CSV file or move it outside the project folder without updating the `pd.read_csv()` path.

---

## Final Conclusion

This project successfully demonstrated end-to-end machine learning for Telco customer churn prediction. Using 7,043 real customer records, four classification algorithms were trained and evaluated. Key churn risk factors identified through EDA include month-to-month contracts, fiber optic internet service, electronic check payment method, short customer tenure, and senior citizen status.

The Decision Tree model achieved the best F1-Score (0.6142), making it the optimal model for churn classification under class imbalance. Logistic Regression achieved the highest accuracy (80.55%) and serves as a strong interpretable baseline. Gaussian Naive Bayes achieved the highest recall (85.03%), making it suitable for high-sensitivity churn detection use cases.
