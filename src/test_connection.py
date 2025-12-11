from config import engine

try:
    conn = engine.connect()
    print("✅ Python successfully connected to MySQL!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:")
    print(e)
