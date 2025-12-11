import pandas as pd
from sqlalchemy import text
from config import engine


def build_application_features():
    """Base features from current application table."""
    with engine.begin() as conn:
        app = pd.read_sql(text("SELECT * FROM stg_application_train"), conn)

    df = app.copy()

    # Basic numeric safety
    df["AMT_INCOME_TOTAL"].replace(0, pd.NA, inplace=True)

    # Core ratios
    df["income_to_credit_ratio"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["annuity_to_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] / 12)
    df["credit_to_goods_ratio"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    # Age and employment
    df["age_years"] = -df["DAYS_BIRTH"] / 365
    df["employment_duration_years"] = -df["DAYS_EMPLOYED"].clip(upper=0) / 365
    df["days_employed_ratio"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]

    # Income per family member
    df["income_per_family_member"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, pd.NA)

    # We keep raw categorical columns for later encoding in the ML pipeline
    return df


def build_bureau_features():
    """Aggregates from external bureau tables."""
    import numpy as np

    with engine.begin() as conn:
        bureau = pd.read_sql(text("SELECT * FROM stg_bureau"), conn)
        bb = pd.read_sql(text("SELECT * FROM stg_bureau_balance"), conn)

    # Bureau-level aggregates per SK_ID_CURR
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_num_loans=("SK_ID_BUREAU", "count"),
        bureau_active_loans=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_closed_loans=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        bureau_consumer_loans=("CREDIT_TYPE", lambda x: (x == "Consumer credit").sum()),
        bureau_mortgage_loans=("CREDIT_TYPE", lambda x: (x == "Mortgage").sum()),
        bureau_total_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bureau_total_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_total_limit_sum=("AMT_CREDIT_SUM_LIMIT", "sum"),
        bureau_max_overdue_days=("CREDIT_DAY_OVERDUE", "max"),
        bureau_avg_days_credit=("DAYS_CREDIT", "mean"),
        bureau_min_days_credit=("DAYS_CREDIT", "min"),
    ).reset_index()

    # SAFE zero-to-null replacement (no recursion)
    bureau_agg["bureau_total_limit_sum"] = bureau_agg["bureau_total_limit_sum"].where(
        bureau_agg["bureau_total_limit_sum"] != 0, np.nan
    )

    # Utilisation ratio
    bureau_agg["bureau_debt_to_limit_ratio"] = (
        bureau_agg["bureau_total_debt_sum"] / bureau_agg["bureau_total_limit_sum"]
    )

    # Bureau_balance processing
    bb = bb.merge(bureau[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU", how="left")

    bad_statuses = ["1", "2", "3", "4", "5"]
    bb["is_bad_status"] = bb["STATUS"].isin(bad_statuses).astype(int)

    bb_agg = bb.groupby("SK_ID_CURR").agg(
        bb_months_on_books=("MONTHS_BALANCE", "count"),
        bb_bad_months=("is_bad_status", "sum"),
        bb_last_month=("MONTHS_BALANCE", "max"),
    ).reset_index()

    bb_agg["bb_bad_month_share"] = (
        bb_agg["bb_bad_months"] / bb_agg["bb_months_on_books"]
    )

    bureau_full = bureau_agg.merge(bb_agg, on="SK_ID_CURR", how="left")

    return bureau_full



def build_prev_installment_features():
    """Previous applications + installments aggregated per current customer."""
    with engine.begin() as conn:
        prev = pd.read_sql(text("SELECT * FROM stg_previous_application"), conn)
        inst = pd.read_sql(text("SELECT * FROM stg_installments_payments"), conn)

    # Previous application aggregates at SK_ID_CURR
    prev_agg = prev.groupby("SK_ID_CURR").agg(
        prev_num_applications=("SK_ID_PREV", "count"),
        prev_num_approved=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        prev_num_refused=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        prev_avg_amt_credit=("AMT_CREDIT", "mean"),
        prev_max_amt_credit=("AMT_CREDIT", "max"),
        prev_cash_loans=("NAME_CONTRACT_TYPE", lambda x: (x == "Cash loans").sum()),
        prev_consumer_loans=("NAME_CONTRACT_TYPE", lambda x: (x == "Consumer loans").sum()),
        prev_days_decision_last=("DAYS_DECISION", "min"),
    ).reset_index()

    prev_agg["prev_approval_rate"] = (
        prev_agg["prev_num_approved"] / prev_agg["prev_num_applications"].replace(0, pd.NA)
    )

    # Installments: aggregate per SK_ID_PREV, then bring to SK_ID_CURR via previous_application
    inst["days_late"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["days_early"] = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    inst_prev_agg = inst.groupby("SK_ID_PREV").agg(
        inst_total_installments=("NUM_INSTALMENT_NUMBER", "count"),
        inst_paid_total=("AMT_PAYMENT", "sum"),
        inst_instalment_total=("AMT_INSTALMENT", "sum"),
        inst_missed_installments=("AMT_PAYMENT", lambda x: (x == 0).sum()),
        inst_avg_payment_delay_days=("days_late", "mean"),
        inst_max_payment_delay_days=("days_late", "max"),
        inst_share_early_payments=("days_early", lambda x: (x > 0).mean()),
    ).reset_index()

    inst_prev_agg["inst_missed_installment_ratio"] = (
        inst_prev_agg["inst_missed_installments"] / inst_prev_agg["inst_total_installments"].replace(0, pd.NA)
    )

    prev_with_inst = prev.merge(inst_prev_agg, on="SK_ID_PREV", how="left")

    inst_curr_agg = prev_with_inst.groupby("SK_ID_CURR").agg(
        inst_total_installments=("inst_total_installments", "sum"),
        inst_missed_installments=("inst_missed_installments", "sum"),
        inst_missed_installment_ratio=("inst_missed_installment_ratio", "mean"),
        inst_avg_payment_delay_days=("inst_avg_payment_delay_days", "mean"),
        inst_max_payment_delay_days=("inst_max_payment_delay_days", "max"),
        inst_share_early_payments=("inst_share_early_payments", "mean"),
    ).reset_index()

    prev_full = prev_agg.merge(inst_curr_agg, on="SK_ID_CURR", how="left")

    return prev_full


def build_pos_features():
    """POS_CASH_balance aggregated per current customer via SK_ID_PREV."""
    with engine.begin() as conn:
        pos = pd.read_sql(text("SELECT * FROM stg_pos_cash_balance"), conn)
        prev = pd.read_sql(text("SELECT SK_ID_PREV, SK_ID_CURR FROM stg_previous_application"), conn)

    pos_agg_prev = pos.groupby("SK_ID_PREV").agg(
        pos_months_on_books=("MONTHS_BALANCE", "count"),
        pos_max_dpd=("SK_DPD", "max"),
        pos_avg_dpd=("SK_DPD", "mean"),
        pos_bad_status_months=("SK_DPD", lambda x: (x > 0).sum()),
        pos_active_flag=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").any()),
        pos_completed_flag=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").any()),
    ).reset_index()

    pos_merged = pos_agg_prev.merge(prev, on="SK_ID_PREV", how="left")

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


def build_credit_card_features():
    """Credit card behaviour features per customer."""
    import numpy as np

    with engine.begin() as conn:
        cc = pd.read_sql(text("SELECT * FROM stg_credit_card_balance"), conn)
        prev = pd.read_sql(text("SELECT SK_ID_PREV, SK_ID_CURR FROM stg_previous_application"), conn)

    # SAFE utilization calculation (no pandas replace)
    cc["AMT_CREDIT_LIMIT_ACTUAL"] = cc["AMT_CREDIT_LIMIT_ACTUAL"].where(
        cc["AMT_CREDIT_LIMIT_ACTUAL"] != 0, np.nan
    )
    cc["utilization"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]

    # Aggregate at SK_ID_PREV level
    cc_prev_agg = cc.groupby("SK_ID_PREV").agg(
        cc_months_on_books=("MONTHS_BALANCE", "count"),
        cc_avg_utilization=("utilization", "mean"),
        cc_max_utilization=("utilization", "max"),
        cc_overlimit_months=("AMT_BALANCE", lambda x: (x < 0).sum()),
        cc_total_drawings=("AMT_DRAWINGS_CURRENT", "sum"),
        cc_total_drawings_atm=("AMT_DRAWINGS_ATM_CURRENT", "sum"),
        cc_min_payment_ratio=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
    ).reset_index()

    # Map to SK_ID_CURR
    cc_merged = cc_prev_agg.merge(prev, on="SK_ID_PREV", how="left")

    cc_curr_agg = cc_merged.groupby("SK_ID_CURR").agg(
        cc_total_cards=("SK_ID_PREV", "count"),
        cc_months_on_books=("cc_months_on_books", "sum"),
        cc_avg_utilization=("cc_avg_utilization", "mean"),
        cc_max_utilization=("cc_max_utilization", "max"),
        cc_overlimit_months=("cc_overlimit_months", "sum"),
        cc_total_drawings=("cc_total_drawings", "sum"),
        cc_total_drawings_atm=("cc_total_drawings_atm", "sum"),
    ).reset_index()

    # Cash advance share (safe)
    cc_curr_agg["cc_cash_advance_share"] = (
        cc_curr_agg["cc_total_drawings_atm"] /
        cc_curr_agg["cc_total_drawings"].where(cc_curr_agg["cc_total_drawings"] != 0, np.nan)
    )

    return cc_curr_agg



def build_features_master():
    print("Building application-level features")
    app_df = build_application_features()

    print("Building bureau features")
    bureau_df = build_bureau_features()

    print("Building previous application + installment features")
    prev_inst_df = build_prev_installment_features()

    print("Building POS features")
    pos_df = build_pos_features()

    print("Building credit card features")
    cc_df = build_credit_card_features()

    print("Merging all feature blocks into features_master_train")

    df = app_df.merge(bureau_df, on="SK_ID_CURR", how="left")
    df = df.merge(prev_inst_df, on="SK_ID_CURR", how="left")
    df = df.merge(pos_df, on="SK_ID_CURR", how="left")
    df = df.merge(cc_df, on="SK_ID_CURR", how="left")

    print("Final feature dataframe shape:", df.shape)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS features_master_train"))
        df.to_sql("features_master_train", conn, if_exists="replace", index=False)

    print("features_master_train table created successfully")


if __name__ == "__main__":
    build_features_master()
