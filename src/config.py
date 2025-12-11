from sqlalchemy import create_engine

DB_URI = "mysql+pymysql://root:3690@localhost:3306/credit_risk"
engine = create_engine(DB_URI)
