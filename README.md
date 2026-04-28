B2B Invoice Payment Behaviour Segmentation & Late Payment Predictor

> **Course:** Predictive Analytics — Academic Year 2025-26
>
> Project Overview

In B2B (business-to-business) transactions, companies issue invoices to clients and expect payment within an agreed period. Late payments disrupt cash flow and financial planning — yet almost no company has an ML-based early warning system for this.


This project addresses:
- **Segmenting business customers** by payment behaviour patterns using K-Means and hierarchical clustering
- **Predicting whether an invoice will be paid late** before its due date using an XGBoost classifier
- **Deploying the solution** as an interactive Streamlit web application

This README documents **Stage 3: Data Preprocessing & Cleaning** — the foundation of the entire pipeline.

---

# Stage 3: Preprocessing Pipeline
```
┌──────────────────────────────────────┐
│  Raw Dataset                         │
│  Dataset b2b_invoice_payment.csv     │
│  45,839 rows × 29 columns            │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 1 — Data Loading & Exploration │
│  Shape, dtypes, missing values,      │
│  class balance check                 │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 2 — Cleaning & Validation      │
│  Duplicates, date parsing,           │
│  business-rule checks                │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 3 — Target Variable Creation   │
│  DelayFlag → target_late_payment     │
│  0 = On-time  |  1 = Late            │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 4 — Outlier Detection          │
│  Boxplot visualisation               │
│  IQR Winsorization (1st–99th pct)    │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 5 — Drop Leakage & Redundant   │
│  8 leakage columns removed           │
│  9 redundant columns removed         │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 6 — Scaling & Encoding         │
│  LabelEncoder (categorical)          │
│  StandardScaler (numeric)            │
│  Fit on train only — no leakage      │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 7 — Stratified Train-Test Split│
│  80% train  /  20% test              │
│  stratify=y  |  random_state=42      │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 8 — Class Imbalance (SMOTE)    │
│  Applied on training set only        │
│  36,671 → 48,152 training rows       │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────┐
│  Step 9 — Leakage Audit              │
│  9-point final verification          │
│  All checks passed ✅                │
└─────────────────┬────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────┐
│                    Model-Ready Output                         │
│  X_train_preprocessed.csv   y_train_preprocessed.csv         │
│  X_test_preprocessed.csv    y_test_preprocessed.csv          │
└──────────────────────────────────────────────────────────────┘
```

---

📁 Files in This Stage

| File | Type | Description |
|---|---|---|
| `Dataset b2b_invoice_payment.csv` | Input | Raw B2B invoice dataset — 45,839 rows × 29 columns |
| `stage3_preprocessing.ipynb` | Notebook | Complete Stage 3 preprocessing pipeline |
| `X_train_preprocessed.csv` | Output | SMOTE-balanced, scaled training features — 48,152 rows |
| `y_train_preprocessed.csv` | Output | Training labels |
| `X_test_preprocessed.csv` | Output | Scaled test features — 9,168 rows (untouched) |
| `y_test_preprocessed.csv` | Output | Test labels |
| `target_distribution.png` | Plot | Class balance before preprocessing |
| `outlier_boxplots.png` | Plot | Boxplot visualisation of key columns |
| `smote_comparison.png` | Plot | Class balance before vs after SMOTE |

---
📊 Dataset Overview

| Property | Detail |
|---|---|
| File | `Dataset b2b_invoice_payment.csv` |
| Total rows | 45,839 invoices |
| Total columns | 29 features |
| Target column | `DelayFlag` → renamed `target_late_payment` |
| Late payments — class 1 | 30,096 rows — 65.7% |
| On-time payments — class 0 | 15,743 rows — 34.3% |
| Missing values | None |
| Duplicate rows | None |

🧰 Tools & Libraries

| Library | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Scaling, encoding, train-test split |
| Imbalanced-learn | SMOTE for class imbalance |
| Matplotlib | Boxplots, bar charts |
| Seaborn | Statistical visualisations |

## ⚙️ Preprocessing Summary

The preprocessing pipeline prepares raw invoice data for machine learning by ensuring data quality, removing leakage, and engineering meaningful features.

Key steps include:

* Data loading and validation
* Target variable verification
* Outlier detection and treatment
* Removal of leakage and redundant features
* Feature scaling and encoding
* Stratified train-test split
* Class imbalance handling using SMOTE

A detailed step-by-step pipeline is provided below.

---

## 🔍 Detailed Preprocessing Pipeline

### Step 1 — Data Loading & Exploration

* Loaded `Dataset b2b_invoice_payment.csv` using Pandas
* Confirmed shape: **45,839 rows × 29 columns**
* Inspected data types and unique value counts
* Checked missing values → **none found**
* Checked duplicate rows → **none found**
* Visualised target class distribution → **~2:1 imbalance**

---

### Step 2 — Data Cleaning & Validation

* Parsed date columns to `datetime` (`%m/%d/%Y`):

  * `Doc_Date`, `Net_Due_Date`, `Posting_Date`, `Clearing_date`
* Validated logical constraints:

  * `Doc_Date ≤ Net_Due_Date`
  * `Posting_Date ≤ Clearing_date`
* Verified no negative values in:

  * `Amount`, `Payment_Term`, `Age_Of_Customer_Months`
* Renamed duplicate column:

  * `Weekday_due.1 → Weekday_due_num`

---

### Step 3 — Target Variable Creation

| Value | Meaning                       | Count  |
| ----- | ----------------------------- | ------ |
| `1`   | Invoice paid late             | 30,096 |
| `0`   | Invoice paid on time or early | 15,743 |

* Verified consistency between `DelayFlag` and `Days_Overdue_Delay`
* Renamed `DelayFlag → target_late_payment`

> ⚠️ `Days_Overdue_Delay` is excluded to prevent data leakage.

---

### Step 4 — Outlier Detection & Treatment

* Visualised outliers using boxplots:

  * `Amount`, `Payment_Term`, `Age_Of_Customer_Year`
* Applied **Winsorization (1st–99th percentile capping)**
* Used capping instead of dropping to preserve full dataset

---

### Step 5 — Drop Leakage & Redundant Columns

#### Leakage Columns Removed

| Column             | Reason                   |
| ------------------ | ------------------------ |
| Days_Overdue_Delay | Directly encodes target  |
| Delay_Bins         | Derived from target      |
| Clearing_date      | Known only after payment |
| Clearing_doc       | Created at clearing time |
| Weekday_clearing   | Post-payment info        |
| Weekday_clearnum   | Derived from clearing    |
| Quarter_clearing   | Post-payment info        |

#### Redundant / ID Columns Removed

| Column                               | Reason                |
| ------------------------------------ | --------------------- |
| Document_No                          | Identifier only       |
| Cust_Num                             | Encoded via behavior  |
| Customer_Name                        | Identifier only       |
| Age_Of_Customer_Months               | Redundant             |
| Amount_Bins                          | Duplicate of Amount   |
| Payment_Term_Bins                    | Duplicate             |
| Customer_Age_Year_Bins               | Duplicate             |
| Weekday_due                          | Duplicate             |
| Doc_Date, Net_Due_Date, Posting_Date | Raw features replaced |

---

### Step 6 — Scaling & Encoding

* Label Encoding:

  * `Payment_Method_description`, `Region`, `City`, `Zipcode`, `Weekday_due_num`
* StandardScaler applied to numeric features

> ⚠️ Fitted on training data only to avoid leakage.

---

### Step 7 — Stratified Train-Test Split

| Split    | Rows   | Late   | On-time |
| -------- | ------ | ------ | ------- |
| Training | 36,671 | 24,076 | 12,595  |
| Test     | 9,168  | 6,020  | 3,148   |

* Used `stratify=y` to maintain class distribution

---

### Step 8 — Class Imbalance Handling (SMOTE)

|         | Before | After  |
| ------- | ------ | ------ |
| Class 0 | 12,595 | 24,076 |
| Class 1 | 24,076 | 24,076 |

* Applied only on training data
* Test set remains untouched

---

### Step 9 — Data Leakage Prevention

| Check                          | Status |
| ------------------------------ | ------ |
| Leakage columns removed        | ✅      |
| Scaler fitted on training only | ✅      |
| SMOTE applied only on training | ✅      |
| Test data untouched            | ✅      |

**All checks passed — no data leakage present.**

## ➡️ Output — Ready for Stage 5: Feature Engineering

```
X_train_preprocessed.csv  ←  48,152 rows — SMOTE-balanced, scaled
y_train_preprocessed.csv  ←  48,152 training labels
X_test_preprocessed.csv   ←   9,168 rows — scaled, untouched
y_test_preprocessed.csv   ←   9,168 test labels
```




## 🤖 Stage 2 — Model Training & Evaluation

> **Stage Owner:** Model Training
> **Input from:** Stage 1 (Preprocessing) — `invoices_clean.csv`, `feature_names.pkl`
> **Output to:** Stage 3 (Deployment/API) — `model.pkl`, `scaler.pkl`, `shap_explainer.pkl`

---

###  Objective

Train, evaluate, and select the best machine learning model to predict whether an invoice payment will be **late (1)** or **on-time (0)** using the pre-processed invoice data delivered by Stage 1.

---



---

###  Input Files (from Stage 1)

| File | Description |
|------|-------------|
| `invoices_clean.csv` | Cleaned and preprocessed invoice dataset (45,839 rows) |
| `feature_names.pkl` | Approved list of 30 feature columns (no leakage) |


### Pipeline Steps

**1. Data Loading**
- Loaded `invoices_clean.csv` (45,839 invoices)
- Loaded `feature_names.pkl` → 30 features selected
- Target variable: `DelayFlag` (1 = Late, 0 = On-Time)

**2. Temporal Train-Test Split**
- Sorted by `Doc_Date` (chronological order)
- Train: oldest 80% → 36,671 records (up to 2015-12-11)
- Test: most recent 20% → 9,168 records
- ✅ No random shuffle — prevents data leakage

**3. Class Imbalance Handling**
- Applied SMOTE on training data only
- Train set after SMOTE: 49,660 samples (balanced classes)
- ✅ Test set untouched

**4. Models Trained**

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline, uses scaled features |
| Random Forest | 200 trees, max depth 10 |
| XGBoost | 200 estimators, learning rate 0.05 |

**5. Evaluation Metrics**
All models evaluated on the held-out test set using: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

---

###  Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.4638 | 0.7384 | 0.1029 | 0.1807 | 0.7744 |
| Random Forest | 0.8652 | 0.9732 | 0.7869 | 0.8702 | 0.9179 |
| **XGBoost ✅** | **0.9443** | **0.9837** | **0.9182** | **0.9498** | **0.9905** |

**🏆 Best Model: XGBoost**
- ROC-AUC of 0.9905 — near-perfect discrimination
- Recall of 91.8% — catches 9 out of 10 late payments
- Precision of 98.4% — almost no false alarms

---

###  SHAP Feature Importance (XGBoost)

Top features driving late payment predictions:

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `Weekday_due.1` | Day of week the payment is due |
| 2 | `order_volume_ratio` | Invoice size vs customer's usual volume |
| 3 | `cust_hist_avg_overdue` | Customer's average historical overdue days |
| 4 | `Weekday_clearnum` | Day of week the invoice was cleared |
| 5 | `Payment_Method_description_enc` | Payment method used by customer |

> **Key insight:** Customer payment history (`cust_hist_avg_overdue`, `cust_hist_late_rate`) and timing features (`Weekday_due.1`, `invoice_dayofweek`) are the strongest predictors of late payment.

---

###  Output Files

| File | Description 
|------|-------------
| `models/model.pkl` | Trained XGBoost model 
| `models/scaler.pkl` | StandardScaler fitted on training data 
| `models/shap_explainer.pkl` | SHAP explainer for predictions 
| `results/metrics.json` | All model evaluation metrics
| `feature_names.pkl` | Feature list 




