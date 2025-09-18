import pandas as pd
import mysql.connector

df = pd.read_csv("troubleshooting_dataset.csv")

# Koneksi ke MySQL (Laragon)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",          
    database="admisi_db"
)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS qa_dataset (
    id INT AUTO_INCREMENT PRIMARY KEY,
    intent VARCHAR(255),
    question TEXT,
    answer TEXT,
    label VARCHAR(255)
)
""")

for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO qa_dataset (intent, question, answer, label)
        VALUES (%s, %s, %s, %s)
    """, (row["intent"], row["question"], row["answer"], row["label"]))

conn.commit()
cursor.close()
conn.close()

print("Data berhasil dimasukkan ke MySQL (Laragon)")
