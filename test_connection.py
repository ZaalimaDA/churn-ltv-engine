import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="telco_churn_db",
    user="churn_admin",
    password="skad5",
    port="5432"
)

cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM customers")
print(cur.fetchone())
tables = cur.fetchall()

print("Tables:")
for table in tables:
    print(table[0])

cur.close()
conn.close()