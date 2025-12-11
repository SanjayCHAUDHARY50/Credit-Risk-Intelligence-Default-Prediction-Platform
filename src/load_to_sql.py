import pandas as pd
from config import engine
import os

DATA_PATH = "../data"
CHUNK_SIZE = 50000  # Safe for large files


def load_csv_to_sql(filename, table_name):
    file_path = os.path.join(DATA_PATH, filename)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {filename} into table {table_name}")

    first_chunk = True

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        chunk.to_sql(
            table_name,
            con=engine,
            if_exists="replace" if first_chunk else "append",
            index=False
        )
        first_chunk = False

    print(f"Finished loading table: {table_name}")


def main():
    files = {
        "application_train.csv": "stg_application_train",
        "application_test.csv": "stg_application_test",
        "bureau.csv": "stg_bureau",
        "bureau_balance.csv": "stg_bureau_balance",
        "POS_CASH_balance.csv": "stg_pos_cash_balance",
        "credit_card_balance.csv": "stg_credit_card_balance",
        "previous_application.csv": "stg_previous_application",
        "installments_payments.csv": "stg_installments_payments",
    }

    print("Starting CSV to MySQL load process")

    for file, table in files.items():
        load_csv_to_sql(file, table)

    print("All staging tables loaded successfully")


if __name__ == "__main__":
    main()
