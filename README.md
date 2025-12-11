Credit Risk Intelligence & Default Prediction Platform
A full end-to-end enterprise credit risk scoring system built using Python, SQL, Machine Learning, and Power BI.
This project implements a complete credit risk analytics platform inspired by real banking workflows.
It covers the entire lifecycle from raw data ingestion to model scoring and business intelligence reporting.

Dataset used:
Home Credit Default Risk — Kaggle Competition
https://www.kaggle.com/competitions/home-credit-default-risk

1. Project Overview
Banks and lending companies must approve loans while minimizing credit losses.
This platform replicates a real enterprise credit scoring system by integrating:
Multi-table data ingestion and preprocessing
SQL-based feature storage
Feature engineering that combines application, credit bureau, and behavioral data
Machine Learning with Probability of Default (PD) prediction
Business risk layers including Risk Buckets and Expected Loss
Power BI dashboards for portfolio-level insights
Kaggle-compatible submission file for hackathon use

The goal is to build a production-style ML pipeline that is both technically sound and business-ready.

2. Architecture and Approach
High-Level Pipeline
Raw Kaggle Data → SQL Staging → Feature Engineering → Feature Store
                                  ↓
                            Model Training
                                  ↓
       ┌──────────────────────────┴──────────────────────────┐
       │                                                     │
 Kaggle Submission (Test PD)                       Power BI Risk Layer (Train PD)

3. Dataset Description

Source: Home Credit Default Risk (Kaggle)
The dataset consists of multiple relational tables:

application_train.csv
Main table containing applicant information and the target variable (default: 0/1).

application_test.csv
Same as training data but without the target; used for final predictions.

bureau.csv
Applicant’s past loans with other financial institutions.

bureau_balance.csv
Monthly status history of bureau loans.

previous_application.csv
Previous loan applications at Home Credit.

installments_payments.csv
Payment history of previous loans.

POS_CASH_balance.csv
Monthly snapshots for POS/Cash loans.

credit_card_balance.csv
Monthly snapshots of credit card balances.

Large raw files were intentionally removed from the GitHub repository to reduce repository size.
Users should download the dataset directly from Kaggle.

4. Feature Engineering
Feature engineering is performed in Python and stored in a MySQL feature store.
Features are aggregated at the applicant level (SK_ID_CURR).

4.1 Application-Level Features
Examples:
income_to_credit_ratio
age_client
employment_length
annuity_to_income_ratio
credit_to_goods_ratio

4.2 Bureau Features
Aggregated from bureau.csv and bureau_balance.csv:
bureau_num_loans
bureau_active_loans
bureau_total_debt_sum
bureau_total_credit_limit_sum
bureau_debt_to_limit_ratio

4.3 Previous Application + Installment Features
num_prev_applications
approval_rate
mean_prev_credit
missed_payment_ratio
on_time_payment_ratio

4.4 POS_CASH Features
pos_num_loans
pos_avg_dpd
pos_max_dpd
pos_status_ratio

4.5 Credit Card Features
cc_avg_utilization
cc_balance_to_limit_ratio
cc_months_active
cc_avg_payment

All features are merged into:
features_master_train
features_master_test

5. Machine Learning
5.1 Model A (ROC-AUC Optimized)
The first model focuses on achieving the highest discrimination power.
Model: RandomForestClassifier
Data split: 80% train / 20% validation
Metric: ROC-AUC
Validation Results:
ROC-AUC: approximately 0.75
Recall for defaulters: approximately 0.53

Outputs:
Trained model saved with joblib
submission_auc.csv for Kaggle-style evaluation
scores_train and scores_test tables with PD, Risk Bucket, Expected Loss

6. Risk Layer (Business Intelligence)
After PD is generated, the system computes:

6.1 Risk Bucket
Based on PD thresholds:
Low: PD < 5%
Medium: 5% ≤ PD < 20%
High: PD ≥ 20%
(Adjusted in later versions for business use and Power BI visualization.)

6.2 Expected Loss

Calculated as:
expected_loss = PD × LGD × AMT_CREDIT
LGD is assumed as 45%.
These values are stored in MySQL and used for Power BI dashboards.

7. Power BI Dashboards
The following dashboards are created:
Portfolio PD distribution
Risk buckets (Low, Medium, High)
Expected Loss by customer group
Income vs credit analysis
Branch or demographic-level risk segmentation
Feature importance and drivers of risk
These dashboards demonstrate the practical business value of the ML system.

8. Project Structure
Example structure:
project_root/
│
├── src/
│   ├── load_to_sql.py
│   ├── build_features_master.py
│   ├── train_and_submit_auc.py
│   ├── utils/
│   └── ...
│
├── models/
│   └── risk_model_auc.joblib
│
├── output/
│   └── submission_auc.csv
│
├── sql/
│   └── schema_definitions.sql
│
├── README.md
└── requirements.txt


Raw dataset files are not included in the repository to keep file size small.

9. How to Run the Project
Step 1: Download Dataset
Download all CSV files from:
https://www.kaggle.com/competitions/home-credit-default-risk
Place them in a local directory such as ./data/.

Step 2: Load into MySQL
Run:
python load_to_sql.py

Step 3: Create Feature Store
python build_features_master.py

Step 4: Train Model A (ROC-AUC Optimized)
python train_and_submit_auc.py


Outputs:
submission_auc.csv
scores_train (MySQL)
scores_test (MySQL)

Step 5: Build Power BI Dashboard

Connect Power BI to MySQL and load:
scores_train
features_master_train

10. Results Summary
Model A performance:
ROC-AUC ≈ 0.75
Defaulter Recall ≈ 0.53
Expected Loss and Risk Buckets computed correctly

All test applicants scored and exported to submission file
The project provides:
A hackathon-ready ML model
A resume-ready enterprise analytics case study
A complete SQL + Python + BI solution

11. Notes
Raw dataset files are not included due to size.
Installation requires Python 3.10, MySQL 8.0, and Power BI Desktop.
Future extensions can include Model B (High Recall) and Model C (Balanced Business Model).

12. License
This repository is for educational and portfolio use.
You may modify and extend the code for personal or academic purposes.
