import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="telco_churn_db",
    user="churn_admin",
    password="santhoz303",
    port="5432"
)

cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM customers")
print(cur.fetchone())

# Correct way to list tables
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
""")
tables = cur.fetchall()

print("Tables:")
for table in tables:
    print(table[0])

cur.close()
conn.close()