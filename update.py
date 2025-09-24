import pandas as pd
import mysql.connector

df = pd.read_csv("troubleshooting_dataset.csv")
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="admisi_db"
)
cursor = conn.cursor()

# Kosongkan tabel lama biar bersih
cursor.execute("TRUNCATE TABLE qa_dataset")

# Insert data baru
for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO qa_dataset (intent, question, answer, label)
        VALUES (%s, %s, %s, %s)
        """,
        (row["intent"], row["question"], row["answer"], row["label"])
    )

conn.commit()
cursor.close()
conn.close()

print("Database berhasil diperbarui dari CSV!")
