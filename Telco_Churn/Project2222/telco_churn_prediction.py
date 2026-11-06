# ==============================================================================
# PROJECT OBJECTIVES:
# 1. Identify Key Churn Factors: Uncover how tenure, contract type, charges,
#    and services influence customer churn through data exploration.
# 2. Ensure Data Quality: Clean dataset integrity by imputing missing values with medians and removing extreme outliers using IQR.
# 3. Prepare ML-Ready Features: Transform categorical variables into numerical form (Label & One-Hot Encoding) and
#    normalize feature scales using MinMaxScaler.
# 4. Predict Churn Outcomes: Train and optimize diverse classification algorithms (Logistic Regression,
#    KNN, Naive Bayes, Decision Tree) to accurately predict customer churn.
# 5. Benchmark Model Performance: Compare all four classifiers across Accuracy, Precision, Recall,
#    and F1-Score to determine the best model for deployment.
# ==============================================================================

#================ IMPORTING LIBRARIES ================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#================ LOADING DATASET ================
data = pd.read_csv(r"C:\Users\HP\Desktop\Project2222\Telco_customer_churn.csv")

#================ OBJECTIVE 1: IDENTIFY KEY CHURN FACTORS (EDA) ================

#================ BASIC DATASET INFORMATION ================
print(data.head())
print(data.shape)
print(data.columns)

#================ DATASET INFORMATION ================
print(data.info())

#================ CONVERTING TOTAL CHARGES TO NUMERIC ================
# Total Charges is stored as text; blank strings (new customers) become NaN
data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")

#================ STATISTICAL SUMMARY ================
print(data.describe())

#================ CHECKING MISSING VALUES ================
print(data.isnull().sum())

#================ CHECKING DUPLICATE VALUES ================
print(data.duplicated().sum())

#================ CHECKING UNIQUE VALUES ================
print(data.nunique())

#================ TARGET VARIABLE DISTRIBUTION ================
print(data["Churn Value"].value_counts())
print(data["Churn Value"].value_counts(normalize=True) * 100)

#================ DROPPING NON-PREDICTIVE / LEAKAGE COLUMNS ================
# Identifiers, constant columns, location columns and columns that directly reveal churn
data = data.drop(["CustomerID", "Count", "Country", "State", "City", "Zip Code",
                  "Lat Long", "Latitude", "Longitude",
                  "Churn Label", "Churn Score", "Churn Reason"], axis=1)

#================ NUMERICAL AND CATEGORICAL COLUMNS ================
print("Numerical Columns:")
print(data.select_dtypes(include=["int64", "float64"]).columns)
print("Categorical Columns:")
print(data.select_dtypes(include=["object", "category", "string"]).columns)

#================ CATEGORICAL COLUMN DISTRIBUTION ================
print(data["Gender"].value_counts())
print(data["Contract"].value_counts())
print(data["Internet Service"].value_counts())
print(data["Payment Method"].value_counts())
print(data["Churn Value"].value_counts())

#================ TARGET VARIABLE VISUALIZATION ================
sns.countplot(x="Churn Value", data=data)
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

#================ NUMERICAL FEATURE DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["Tenure Months"], bins=20, kde=True)
plt.title("Tenure Months Distribution")
plt.xlabel("Tenure Months")
plt.ylabel("Number of Customers")
plt.show()

#================ MONTHLY CHARGES DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["Monthly Charges"], bins=20, kde=True)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.show()

#================ TOTAL CHARGES DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["Total Charges"], bins=20, kde=True)
plt.title("Total Charges Distribution")
plt.xlabel("Total Charges")
plt.ylabel("Number of Customers")
plt.show()

#================ CLTV DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.histplot(data["CLTV"], bins=20, kde=True)
plt.title("CLTV Distribution")
plt.xlabel("CLTV")
plt.ylabel("Number of Customers")
plt.show()

#================ GENDER DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", data=data)
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()

#================ SENIOR CITIZEN DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Senior Citizen", data=data)
plt.title("Senior Citizen Distribution")
plt.xlabel("Senior Citizen")
plt.ylabel("Number of Customers")
plt.show()

#================ PARTNER DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Partner", data=data)
plt.title("Partner Distribution")
plt.xlabel("Partner")
plt.ylabel("Number of Customers")
plt.show()

#================ DEPENDENTS DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Dependents", data=data)
plt.title("Dependents Distribution")
plt.xlabel("Dependents")
plt.ylabel("Number of Customers")
plt.show()

#================ PHONE SERVICE DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Phone Service", data=data)
plt.title("Phone Service Distribution")
plt.xlabel("Phone Service")
plt.ylabel("Number of Customers")
plt.show()

#================ MULTIPLE LINES DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Multiple Lines", data=data)
plt.title("Multiple Lines Distribution")
plt.xlabel("Multiple Lines")
plt.ylabel("Number of Customers")
plt.show()

#================ INTERNET SERVICE DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Internet Service", data=data)
plt.title("Internet Service Distribution")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.show()

#================ ONLINE SECURITY DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Online Security", data=data)
plt.title("Online Security Distribution")
plt.xlabel("Online Security")
plt.ylabel("Number of Customers")
plt.show()

#================ ONLINE BACKUP DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Online Backup", data=data)
plt.title("Online Backup Distribution")
plt.xlabel("Online Backup")
plt.ylabel("Number of Customers")
plt.show()

#================ DEVICE PROTECTION DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Device Protection", data=data)
plt.title("Device Protection Distribution")
plt.xlabel("Device Protection")
plt.ylabel("Number of Customers")
plt.show()

#================ TECH SUPPORT DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Tech Support", data=data)
plt.title("Tech Support Distribution")
plt.xlabel("Tech Support")
plt.ylabel("Number of Customers")
plt.show()

#================ STREAMING TV DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Streaming TV", data=data)
plt.title("Streaming TV Distribution")
plt.xlabel("Streaming TV")
plt.ylabel("Number of Customers")
plt.show()

#================ STREAMING MOVIES DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Streaming Movies", data=data)
plt.title("Streaming Movies Distribution")
plt.xlabel("Streaming Movies")
plt.ylabel("Number of Customers")
plt.show()

#================ CONTRACT DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Contract", data=data)
plt.title("Contract Distribution")
plt.xlabel("Contract")
plt.ylabel("Number of Customers")
plt.show()

#================ PAPERLESS BILLING DISTRIBUTION ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Paperless Billing", data=data)
plt.title("Paperless Billing Distribution")
plt.xlabel("Paperless Billing")
plt.ylabel("Number of Customers")
plt.show()

#================ PAYMENT METHOD DISTRIBUTION ================
plt.figure(figsize=(10, 5))
sns.countplot(x="Payment Method", data=data)
plt.title("Payment Method Distribution")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.show()

#================ GENDER VS CHURN ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", hue="Churn Value", data=data)
plt.title("Gender vs Churn")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()

#================ CONTRACT VS CHURN ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Contract", hue="Churn Value", data=data)
plt.title("Contract vs Churn")
plt.xlabel("Contract")
plt.ylabel("Number of Customers")
plt.show()

#================ INTERNET SERVICE VS CHURN ================
plt.figure(figsize=(8, 5))
sns.countplot(x="Internet Service", hue="Churn Value", data=data)
plt.title("Internet Service vs Churn")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")
plt.show()

#================ PAYMENT METHOD VS CHURN ================
plt.figure(figsize=(12, 6))
sns.countplot(x="Payment Method", hue="Churn Value", data=data)
plt.title("Payment Method vs Churn")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.show()

#================ TENURE MONTHS VS CHURN (BOXPLOT) ================
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn Value", y="Tenure Months", data=data)
plt.title("Tenure Months vs Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Tenure Months")
plt.show()

#================ MONTHLY CHARGES VS CHURN (BOXPLOT) ================
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn Value", y="Monthly Charges", data=data)
plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges")
plt.show()

#================ TENURE MONTHS VS MONTHLY CHARGES (SCATTERPLOT) ================
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Tenure Months", y="Monthly Charges", hue="Churn Value", data=data, alpha=0.7)
plt.title("Tenure Months vs Monthly Charges by Churn")
plt.xlabel("Tenure Months")
plt.ylabel("Monthly Charges")
plt.show()

#================ TENURE MONTHS VS TOTAL CHARGES (SCATTERPLOT) ================
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Tenure Months", y="Total Charges", hue="Churn Value", data=data, alpha=0.7)
plt.title("Tenure Months vs Total Charges by Churn")
plt.xlabel("Tenure Months")
plt.ylabel("Total Charges")
plt.show()

#================ CORRELATION ANALYSIS ================
corr = data.select_dtypes(include="number").corr()
print(corr)

plt.figure(figsize=(16, 10))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

#================ OBJECTIVE 2: ENSURE DATA QUALITY (MISSING VALUES & OUTLIERS) ================

#================ CHECKING MISSING VALUES ================
print(data.isnull().sum())
print("Total missing values:", data.isnull().sum().sum())

#================ CHECKING DUPLICATE ROWS ================
print("Total duplicate rows:", data.duplicated().sum())

#================ HANDLING MISSING VALUES ================
missing_columns = {
    "Total Charges": data["Total Charges"].median()
}
data.fillna(missing_columns, inplace=True)

#================ VERIFYING MISSING VALUES ================
print(data.isnull().sum())

#================ CHECKING OUTLIERS ================
numerical_columns = data.select_dtypes(include="number").columns.drop(["Churn Value"], errors="ignore")

for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_limit) | (data[column] > upper_limit)]
    print(column, ":", len(outliers))

#================ OUTLIER VISUALIZATION (BOXPLOT) ================
plt.figure(figsize=(8, 4))
sns.boxplot(x=data["Monthly Charges"])
plt.title("Monthly Charges Boxplot (Outlier Detection)")
plt.xlabel("Monthly Charges")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=data["Total Charges"])
plt.title("Total Charges Boxplot (Outlier Detection)")
plt.xlabel("Total Charges")
plt.show()

#================ REMOVING OUTLIERS ================
for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    data = data[
        (data[column] >= lower_limit) &
        (data[column] <= upper_limit)
    ]

print("Shape after removing outliers:", data.shape)

#================ OBJECTIVE 3: PREPARE ML-READY FEATURES (ENCODING & SCALING) ================

#================ ENCODING BINARY COLUMNS LABEL ENCODING ================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
binary_columns = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]
for column in binary_columns:
    data[column] = le.fit_transform(data[column])

#================ ENCODING MULTI-CATEGORY COLUMNS ONE HOT ENCODING================
data = pd.get_dummies(data, columns=["Multiple Lines", "Internet Service", "Online Security",
                                     "Online Backup", "Device Protection", "Tech Support",
                                     "Streaming TV", "Streaming Movies", "Contract",
                                     "Payment Method"], dtype='int')

#================ SEPARATING FEATURES AND TARGET ================
X = data.drop(["Churn Value"], axis=1)
y = data["Churn Value"]

#================ TRAIN TEST SPLIT ================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)

#================ SCALING NUMERICAL COLUMNS ================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = X_train.copy()
X_test = X_test.copy()

numerical_columns = X_train.select_dtypes(include="number").columns

X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

#================ OBJECTIVE 4: PREDICT CHURN OUTCOMES (MODEL TRAINING) ================

#================ TRAINING LOGISTIC REGRESSION ================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#================ EVALUATING LOGISTIC REGRESSION ================
print("================ Logistic Regression ================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("=====================================================")

#================ LOGISTIC REGRESSION CONFUSION MATRIX ================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#================ TRAINING KNN ================
from sklearn.neighbors import KNeighborsClassifier

for k in range(1, 21):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    y_pred_k = model_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_k)
    print("K =", k, "Accuracy =", accuracy)

#================Suitable K ================
for k in range(1,21):
    model_knn = KNeighborsClassifier(n_neighbors=k,weights="distance",metric="minkowski")
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)

    print(
        "K =", k,
        "| Accuracy =", round(accuracy_score(y_test, y_pred), 3),
        "| Precision =", round(precision_score(y_test, y_pred), 3),
        "| Recall =", round(recall_score(y_test, y_pred), 3),
        "| F1 Score =", round(f1_score(y_test, y_pred), 3)
    )

#Conclusion-> As K increases, accuracy generally rises and the scores flatten out around K = 15 to 16 (F1-score ~0.568). Very small K (1 to 2) gives the weakest scores.

#choose k as 15
model_knn = KNeighborsClassifier(n_neighbors=15)
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

#================ EVALUATING KNN ================
print("================ KNN ================")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precision:", precision_score(y_test, y_pred_knn))
print("Recall:", recall_score(y_test, y_pred_knn))
print("F1 Score:", f1_score(y_test, y_pred_knn))
print("=====================================================")

#================ KNN CONFUSION MATRIX ================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt="d", cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#================ TRAINING NAIVE BAYES ================
from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

#================ EVALUATING NAIVE BAYES ================
print("================ NAIVE BAYES ================")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb))
print("Recall:", recall_score(y_test, y_pred_nb))
print("F1 Score:", f1_score(y_test, y_pred_nb))
print("=====================================================")

#================ NAIVE BAYES CONFUSION MATRIX ================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#================ TRAINING DECISION TREE ================
from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

#================ EVALUATING DECISION TREE ================
print("================ Decision Tree ================")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1 Score:", f1_score(y_test, y_pred_dt))
print("==============================================")

#================ DECISION TREE CONFUSION MATRIX ================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#================ OBJECTIVE 5: BENCHMARK MODEL PERFORMANCE (MODEL COMPARISON) ================

#================ MODEL COMPARISON ================
y_pred_lr = model.predict(X_test)

comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_nb),
        accuracy_score(y_test, y_pred_dt)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_nb),
        precision_score(y_test, y_pred_dt)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_nb),
        recall_score(y_test, y_pred_dt)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_nb),
        f1_score(y_test, y_pred_dt)
    ]
})

print("================ MODEL COMPARISON ================")
print(comparison)

#================ MODEL ACCURACY COMPARISON ================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=comparison)
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()

#================ MODEL F1 SCORE COMPARISON ================
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="F1 Score", data=comparison)
plt.title("Model F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.show()

#================ FINAL CONCLUSION ================
# Conclusion -> Logistic Regression delivered the best overall performance with 80.5% accuracy and an F1-score of 0.611.
# KNN (K=15) also performed well (~78.3% accuracy and 0.597 F1-score), while Naive Bayes achieved the highest recall (~82.6%) but with low precision (~47.1%).
# Decision Tree performed the weakest (~72.1% accuracy, 0.484 F1-score).
# Therefore, Logistic Regression is selected as the optimal model for customer churn prediction on this dataset.
