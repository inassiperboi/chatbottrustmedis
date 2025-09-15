import random
import numpy as np
import pandas as pd
import re, string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

CONFIDENCE_THRESHOLD = 0.65  

df = pd.read_csv("troubleshooting_dataset.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    return text.strip()

df["clean_question"] = df["question"].apply(clean_text)

intent_encoder = LabelEncoder()
df["intent_encoded"] = intent_encoder.fit_transform(df["intent"])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(df["clean_question"].tolist())
y_train = df["intent_encoded"].tolist()

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Hybrid Chatbot Function
def hybrid_chatbot(user_input):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    
    pred_proba = model.predict_proba(vectorized)[0]
    confidence = max(pred_proba)
    pred_intent = model.predict(vectorized)[0]
    intent = intent_encoder.inverse_transform([pred_intent])[0]

    # Debug info (bisa dimatikan kalau sudah stabil)
    print(f"[DEBUG] Predicted intent: {intent}, Confidence: {confidence:.2f}")

    if confidence >= CONFIDENCE_THRESHOLD:
        # Ambil jawaban dari dataset
        answers = df[df["intent"] == intent]["answer"].unique().tolist()
        if answers:
            return f"{random.choice(answers)} (Intent: {intent}, conf={confidence:.2f})"
        else:
            return f"Maaf, belum ada jawaban untuk intent '{intent}' (conf={confidence:.2f})"
    else:
        # Fallback dengan semantic similarity
        all_vectors = vectorizer.transform(df["clean_question"].tolist())
        sim_scores = cosine_similarity(vectorized, all_vectors).flatten()
        best_idx = np.argmax(sim_scores)
        best_score = sim_scores[best_idx]

        print(f"[DEBUG] Fallback similarity score: {best_score:.2f}")

        if best_score > 0.3:
            return f"{df.iloc[best_idx]['answer']} (Mirip dengan pertanyaan: '{df.iloc[best_idx]['question']}')"
        else:
            return "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."

if __name__ == "__main__":
    print("=== Chatbot Hybrid Test ===")
    while True:
        user_inp = input("Anda : ")
        if user_inp.lower() in ["exit", "quit", "keluar"]:
            print("Chatbot : Sampai jumpa!")
            break
        response = hybrid_chatbot(user_inp)
        print("Chatbot :", response)
