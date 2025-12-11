# train_and_submit_auc.py
# Run from credit_risk_platform/src
# Usage: python train_and_submit_auc.py
#
# Important: requires features_master_train already created in MySQL.
# The script will build features_master_test from stg_application_test and other stg_ tables.

import os
import numpy as np
import pandas as pd
from sqlalchemy import text
from config import engine
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, recall_score
import joblib

# ---------- USER TUNABLE ----------
LGD = 0.45   # Loss-Given-Default assumption for expected loss
RISK_BUCKETS = [(0.05, "Low"), (0.20, "Medium"), (1.01, "High")]  # PD thresholds (exclusive of lower)
MODEL_PATH = "../models/risk_model_auc.joblib"
PIPE_PATH = "../models/preprocessor_auc.joblib"
SUBMISSION_CSV = "../submission_auc.csv"
# -----------------------------------

def risk_bucket_from_pd(pd_series):
    def _bucket(p):
        for thr, name in RISK_BUCKETS:
            if p < thr:
                return name
        return "High"
    return pd_series.apply(_bucket)

# Reusable feature-block builders adapted to use a table_name param for application table
def read_table(table_name):
    return pd.read_sql(text(f"SELECT * FROM {table_name}"), con=engine)

def build_application_block(app_table="stg_application_train"):
    with engine.begin() as conn:
        app = pd.read_sql(text(f"SELECT * FROM {app_table}"), conn)

    df = app.copy()
    df["AMT_INCOME_TOTAL"].replace(0, np.nan, inplace=True)
    df["income_to_credit_ratio"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["annuity_to_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] / 12)
    df["credit_to_goods_ratio"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["age_years"] = -df["DAYS_BIRTH"] / 365
    df["employment_duration_years"] = (-df["DAYS_EMPLOYED"].clip(upper=0)) / 365
    df["income_per_family_member"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    return df

def build_bureau_block():
    with engine.begin() as conn:
        bureau = pd.read_sql(text("SELECT * FROM stg_bureau"), conn)
        bb = pd.read_sql(text("SELECT * FROM stg_bureau_balance"), conn)

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_num_loans=("SK_ID_BUREAU", "count"),
        bureau_active_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_closed_loans=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        bureau_total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_total_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_total_limit_sum=("AMT_CREDIT_SUM_LIMIT", "sum"),
        bureau_max_overdue_days=("CREDIT_DAY_OVERDUE", "max"),
        bureau_avg_days_credit=("DAYS_CREDIT", "mean"),
    ).reset_index()

    bureau_agg["bureau_total_limit_sum"] = bureau_agg["bureau_total_limit_sum"].where(
        bureau_agg["bureau_total_limit_sum"] != 0, np.nan
    )
    bureau_agg["bureau_debt_to_limit_ratio"] = (
        bureau_agg["bureau_total_debt_sum"] / bureau_agg["bureau_total_limit_sum"]
    )

    bb = bb.merge(bureau[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU", how="left")
    bad_statuses = ["1","2","3","4","5"]
    bb["is_bad_status"] = bb["STATUS"].isin(bad_statuses).astype(int)
    bb_agg = bb.groupby("SK_ID_CURR").agg(
        bb_months_on_books=("MONTHS_BALANCE", "count"),
        bb_bad_months=("is_bad_status", "sum"),
        bb_last_month=("MONTHS_BALANCE", "max"),
    ).reset_index()
    bb_agg["bb_bad_month_share"] = bb_agg["bb_bad_months"] / bb_agg["bb_months_on_books"].replace(0, np.nan)

    return bureau_agg.merge(bb_agg, on="SK_ID_CURR", how="left")

def build_prev_inst_block():
    with engine.begin() as conn:
        prev = pd.read_sql(text("SELECT * FROM stg_previous_application"), conn)
        inst = pd.read_sql(text("SELECT * FROM stg_installments_payments"), conn)

    prev_agg = prev.groupby("SK_ID_CURR").agg(
        prev_num_applications=("SK_ID_PREV", "count"),
        prev_num_approved=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        prev_num_refused=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        prev_avg_amt_credit=("AMT_CREDIT", "mean"),
        prev_max_amt_credit=("AMT_CREDIT", "max"),
    ).reset_index()
    prev_agg["prev_approval_rate"] = prev_agg["prev_num_approved"] / prev_agg["prev_num_applications"].replace(0, np.nan)

    inst["days_late"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst_prev_agg = inst.groupby("SK_ID_PREV").agg(
        inst_total_installments=("NUM_INSTALMENT_NUMBER", "count"),
        inst_paid_total=("AMT_PAYMENT", "sum"),
        inst_instalment_total=("AMT_INSTALMENT", "sum"),
        inst_missed_installments=("AMT_PAYMENT", lambda x: (x == 0).sum()),
        inst_avg_payment_delay_days=("days_late", "mean"),
        inst_max_payment_delay_days=("days_late", "max"),
    ).reset_index()
    inst_prev_agg["inst_missed_installment_ratio"] = inst_prev_agg["inst_missed_installments"] / inst_prev_agg["inst_total_installments"].replace(0, np.nan)

    prev_with_inst = prev.merge(inst_prev_agg, on="SK_ID_PREV", how="left")
    inst_curr_agg = prev_with_inst.groupby("SK_ID_CURR").agg(
        inst_total_installments=("inst_total_installments", "sum"),
        inst_missed_installments=("inst_missed_installments", "sum"),
        inst_missed_installment_ratio=("inst_missed_installment_ratio", "mean"),
        inst_avg_payment_delay_days=("inst_avg_payment_delay_days", "mean"),
        inst_max_payment_delay_days=("inst_max_payment_delay_days", "max"),
    ).reset_index()

    return prev_agg.merge(inst_curr_agg, on="SK_ID_CURR", how="left")

def build_pos_block():
    with engine.begin() as conn:
        pos = pd.read_sql(text("SELECT * FROM stg_pos_cash_balance"), conn)
        prev_map = pd.read_sql(text("SELECT SK_ID_PREV, SK_ID_CURR FROM stg_previous_application"), conn)
    pos_agg_prev = pos.groupby("SK_ID_PREV").agg(
        pos_months_on_books=("MONTHS_BALANCE", "count"),
        pos_max_dpd=("SK_DPD", "max"),
        pos_avg_dpd=("SK_DPD", "mean"),
        pos_bad_status_months=("SK_DPD", lambda x: (x > 0).sum()),
        pos_active_flag=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").any()),
        pos_completed_flag=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").any()),
    ).reset_index()
    pos_merged = pos_agg_prev.merge(prev_map, on="SK_ID_PREV", how="left")
    pos_curr_agg = pos_merged.groupby("SK_ID_CURR").agg(
        pos_total_loans=("SK_ID_PREV", "count"),
        pos_active_loans=("pos_active_flag", "sum"),
        pos_completed_loans=("pos_completed_flag", "sum"),
        pos_avg_dpd=("pos_avg_dpd", "mean"),
        pos_max_dpd=("pos_max_dpd", "max"),
        pos_bad_status_months=("pos_bad_status_months", "sum"),
        pos_months_on_books=("pos_months_on_books", "sum"),
    ).reset_index()
    return pos_curr_agg

def build_cc_block():
    with engine.begin() as conn:
        cc = pd.read_sql(text("SELECT * FROM stg_credit_card_balance"), conn)
        prev_map = pd.read_sql(text("SELECT SK_ID_PREV, SK_ID_CURR FROM stg_previous_application"), conn)
    cc["AMT_CREDIT_LIMIT_ACTUAL"] = cc["AMT_CREDIT_LIMIT_ACTUAL"].where(cc["AMT_CREDIT_LIMIT_ACTUAL"] != 0, np.nan)
    cc["utilization"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
    cc_prev_agg = cc.groupby("SK_ID_PREV").agg(
        cc_months_on_books=("MONTHS_BALANCE", "count"),
        cc_avg_utilization=("utilization", "mean"),
        cc_max_utilization=("utilization", "max"),
        cc_overlimit_months=("AMT_BALANCE", lambda x: (x < 0).sum()),
        cc_total_drawings=("AMT_DRAWINGS_CURRENT", "sum"),
        cc_total_drawings_atm=("AMT_DRAWINGS_ATM_CURRENT", "sum"),
        cc_min_payment_ratio=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
    ).reset_index()
    cc_merged = cc_prev_agg.merge(prev_map, on="SK_ID_PREV", how="left")
    cc_curr_agg = cc_merged.groupby("SK_ID_CURR").agg(
        cc_total_cards=("SK_ID_PREV", "count"),
        cc_months_on_books=("cc_months_on_books", "sum"),
        cc_avg_utilization=("cc_avg_utilization", "mean"),
        cc_max_utilization=("cc_max_utilization", "max"),
        cc_overlimit_months=("cc_overlimit_months", "sum"),
        cc_total_drawings=("cc_total_drawings", "sum"),
        cc_total_drawings_atm=("cc_total_drawings_atm", "sum"),
    ).reset_index()
    cc_curr_agg["cc_cash_advance_share"] = cc_curr_agg["cc_total_drawings_atm"] / cc_curr_agg["cc_total_drawings"].where(cc_curr_agg["cc_total_drawings"] != 0, np.nan)
    return cc_curr_agg

def build_features_master_for_application_table(app_table="stg_application_test"):
    # Build the blocks and merge (keeps TEST/ TRAIN usage consistent)
    app_block = build_application_block(app_table)
    bureau_block = build_bureau_block()
    prev_inst_block = build_prev_inst_block()
    pos_block = build_pos_block()
    cc_block = build_cc_block()

    df = app_block.merge(bureau_block, on="SK_ID_CURR", how="left")
    df = df.merge(prev_inst_block, on="SK_ID_CURR", how="left")
    df = df.merge(pos_block, on="SK_ID_CURR", how="left")
    df = df.merge(cc_block, on="SK_ID_CURR", how="left")
    return df

def align_test_to_train_columns(train_X, test_df):
    # Ensure test_df has same columns as train_X (drop unexpected, add missing with NaN)
    missing = [c for c in train_X.columns if c not in test_df.columns]
    extra = [c for c in test_df.columns if c not in train_X.columns]
    if extra:
        test_df = test_df.drop(columns=extra)
    for c in missing:
        test_df[c] = np.nan
    # preserve column order
    test_df = test_df[train_X.columns]
    return test_df

def main():
    # --------------------- load train features ---------------------
    print("Loading features_master_train from MySQL")
    with engine.begin() as conn:
        df_train = pd.read_sql(text("SELECT * FROM features_master_train"), conn)

    print("Train shape:", df_train.shape)

    y = df_train["TARGET"].astype(int)
    X = df_train.drop(columns=["TARGET", "SK_ID_CURR"])

    # Split train/test (internal validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessor and model (optimized for ROC-AUC)
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])


    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    model = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42)

    clf = Pipeline([("preproc", preprocessor), ("model", model)])

    print("Training RandomForest (this may take several minutes)...")
    clf.fit(X_train, y_train)

    # Evaluate on held-out validation set
    print("Predicting validation set")
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_val, val_proba)
    recall = recall_score(y_val, val_pred)
    print(f"Validation ROC-AUC: {auc:.4f}")
    print(f"Validation Recall (@0.5): {recall:.4f}")
    print("Classification report (threshold 0.5):")
    print(classification_report(y_val, val_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, val_pred))

    # Save model and preprocessor
    os.makedirs("../models", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

    # -------------------- Score entire training set and write scores_train --------------------
    print("Scoring entire training set and writing to MySQL as scores_train")
    train_proba = clf.predict_proba(X)[:, 1]
    scores_train = pd.DataFrame({
        "SK_ID_CURR": df_train["SK_ID_CURR"],
        "pd_default": train_proba
    })
    scores_train["risk_bucket"] = risk_bucket_from_pd(scores_train["pd_default"])
    scores_train["expected_loss"] = scores_train["pd_default"] * LGD * df_train["AMT_CREDIT"]
    # write to sql
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scores_train"))
        scores_train.to_sql("scores_train", conn, if_exists="replace", index=False)
    print("scores_train written to MySQL")

    # -------------------- Build features for test (stg_application_test) --------------------
    print("Building test feature table from stg_application_test")
    df_test = build_features_master_for_application_table(app_table="stg_application_test")

    # Align test columns with train features
    X_train_columns = X.columns
    df_test_aligned = align_test_to_train_columns(X, df_test.drop(columns=["SK_ID_CURR"], errors="ignore"))
    # score test
    print("Predicting test PDs")
    test_proba = clf.predict_proba(df_test_aligned)[:, 1]
    submission = pd.DataFrame({
        "SK_ID_CURR": df_test["SK_ID_CURR"],
        "TARGET": test_proba
    })
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Submission CSV written to {SUBMISSION_CSV}")

    # Save scores_test to MySQL
    scores_test = pd.DataFrame({
        "SK_ID_CURR": df_test["SK_ID_CURR"],
        "pd_default": test_proba
    })
    scores_test["risk_bucket"] = risk_bucket_from_pd(scores_test["pd_default"])
    scores_test["expected_loss"] = scores_test["pd_default"] * LGD * df_test.get("AMT_CREDIT", np.nan)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scores_test"))
        scores_test.to_sql("scores_test", conn, if_exists="replace", index=False)
    print("scores_test written to MySQL")

    print("All done. Outputs:")
    print(f" - Model: {MODEL_PATH}")
    print(f" - Submission CSV: {SUBMISSION_CSV}")
    print(" - SQL tables: scores_train, scores_test")

if __name__ == "__main__":
    main()
