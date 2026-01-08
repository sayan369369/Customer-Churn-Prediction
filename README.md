# 📊 Customer Churn Prediction using Machine Learning
This project implements a machine learning–based customer churn prediction system using the Telco Customer Churn dataset.

**Machine Learning Course Project**  
Department of Computer Science  
Ramakrishna Mission Vivekananda Educational & Research Institute  
Belur Math, Howrah, West Bengal

---

##  Problem Statement

Customer churn refers to the loss of existing customers over time. Predicting churn allows telecom companies to take preventive actions, improve customer satisfaction, and reduce revenue loss.

The objective of this project is to **build a supervised machine learning model** that predicts whether a customer will **churn (leave the service)** based on **demographic, billing, and service-related features**.

The project uses the **Telco Customer Churn dataset**, consisting of **7,043 customer records** and **20 features** after removing the identifier column. The target variable **Churn (Yes/No)** is **highly imbalanced**, with a majority of non-churning customers. :contentReference[oaicite:0]{index=0}

---

##  Project Objectives

- Perform data cleaning and preprocessing
- Handle class imbalance in the churn dataset
- Train multiple supervised classification models
- Compare model performance using standard evaluation metrics
- Select and save the best-performing model for future prediction

---

##  Proposed Methodology

The workflow follows these key steps:

1. **Data Cleaning**
   - Removed the `customerID` column
   - Replaced blank values in `TotalCharges` with `0.0`
   - Converted `TotalCharges` to numeric format

2. **Feature Encoding**
   - Label-encoded all categorical features
   - Encoded target variable: `Yes → 1`, `No → 0`

3. **Handling Class Imbalance**
   - Applied **SMOTE (Synthetic Minority Oversampling Technique)** on the training data only

4. **Model Training**
   - Trained five classifiers using default hyperparameters
   - Performed **5-fold cross-validation** on balanced training data

5. **Model Evaluation**
   - Evaluated models on unseen test data using:
     - Accuracy
     - Precision
     - Recall
     - F1-score

6. **Model Saving**
   - Saved the best-performing model and label encoders using `pickle` for future inference :contentReference[oaicite:1]{index=1}

---

##  Dataset Details

- **Dataset:** Telco Customer Churn Dataset
- **Total Records:** 7,043
- **Features Used:** 20
- **Target Variable:** `Churn` (binary)
- **Class Distribution:**
  - No Churn: ~5,174
  - Churn: ~1,869 (minority class)

**Key Numerical Features**
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

**Categorical Features**
- `Contract`
- `InternetService`
- `PaymentMethod`
- Customer service and subscription-related attributes

---

## 📊 Exploratory Data Analysis (EDA)

- Distribution analysis of numerical features
- Boxplots for outlier detection
- Correlation heatmap for numerical variables
- Count plots for categorical variables
- Identification of class imbalance in the target variable :contentReference[oaicite:2]{index=2}

---

##  Machine Learning Models Used

The following models were trained and evaluated:

- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Logistic Regression**
- **Support Vector Classifier (SVC)**

All models were trained on **SMOTE-balanced training data** and evaluated on the original test set.

---

##  Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| Decision Tree | 0.754 | 0.67 | 0.64 | 0.65 |
| Random Forest | 0.771 | 0.70 | 0.67 | 0.68 |
| **XGBoost** | **0.781** | **0.71** | **0.69** | **0.70** |
| Logistic Regression | 0.763 | 0.68 | 0.65 | 0.66 |
| SVC | 0.758 | 0.67 | 0.63 | 0.65 |

 **XGBoost achieved the best overall performance**, offering the most balanced trade-off between precision and recall. :contentReference[oaicite:3]{index=3}

---

## 🏆 Best Model Selection

- **Best Model:** XGBoost Classifier
- **Test Accuracy:** ~78%
- **Reason:** Strong performance on imbalanced data and robustness across cross-validation folds

The trained model and encoders are saved as:

```bash
customer_churn_best_model.pkl
encoders.pkl
