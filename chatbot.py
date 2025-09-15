import random
import numpy as np
import pandas as pd
import re, string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# Konfigurasi
CONFIDENCE_THRESHOLD = 0.65
USE_FALLBACK = True  # dipilih karena ini ambang batas aman

# Load Dataset
df = pd.read_csv("troubleshooting_dataset.csv")  # kolom wajib: question, answer, intent

# Preprocessing + Normalisasi
def normalize_text(text):
    replacements = { #key , value
        "satu sehat": "satusehat",
        "satu-sehat": "satusehat",
        "satu  sehat": "satusehat",
        "bpjs kesehatan": "bpjs",
        "rekam medis": "rm",
        "rj" : "rawat jalan"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def clean_text(text):
    text = str(text).lower() # di huruf kecil kan
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = text.strip() # hapus spasi ekstra
    text = normalize_text(text) # memanggil normalisasi
    return text

df["clean_question"] = df["question"].apply(clean_text)

# Encode intents
intent_encoder = LabelEncoder() #buat mapping intentnya ke numerik
df["intent_encoded"] = intent_encoder.fit_transform(df["intent"])
print("Daftar intent:", intent_encoder.classes_)

# Split Train/Test (80/20)
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["intent_encoded"]
)
# TF-IDF Vectorizer (char n-gram tahan typo)
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=5000)
X_train = vectorizer.fit_transform(train["clean_question"].tolist())
X_test = vectorizer.transform(test["clean_question"].tolist())
y_train = train["intent_encoded"].tolist()
y_test = test["intent_encoded"].tolist()

# Train 3 Model digunakan untuk perbandingan
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n=== EVALUASI MODEL ({name}) ===")
    print(classification_report(y_test, y_pred, target_names=intent_encoder.classes_))

    # Cross-validation (5-fold) biar gak overfit
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"Cross-validation F1 (5-fold): {cv_scores}")
    print(f"Rata-rata: {np.mean(cv_scores):.2f}")

# model default Logistic Regression
default_model = models["Logistic Regression"]

def hybrid_chatbot(user_input, model=default_model):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    if hasattr(model, "predict_proba"):  # LogReg, NB
        pred_proba = model.predict_proba(vectorized)[0]
        confidence = max(pred_proba)
    else:  # SVM tidak punya probas
        confidence = 1.0

    pred_intent = model.predict(vectorized)[0]
    intent = intent_encoder.inverse_transform([pred_intent])[0]

    print(f"[DEBUG] Predicted intent: {intent}, Confidence: {confidence:.2f}")

    if confidence >= CONFIDENCE_THRESHOLD:
        answers = df[df["intent"] == intent]["answer"].unique().tolist()
        if answers:
            response = "Berikut beberapa solusi yang bisa dicoba:\n- " + "\n- ".join(answers)
            response += "\n\nJika masalahmu belum teratasi, silakan hubungi admin."
            return response
        else:
            return f"Maaf, belum ada jawaban untuk intent '{intent}'."
    else:
        if USE_FALLBACK: #kalau chatbot ragu pake semantic similarity
            all_vectors = vectorizer.transform(df["clean_question"].tolist())
            sim_scores = cosine_similarity(vectorized, all_vectors).flatten()
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]

            print(f"[DEBUG] Fallback similarity score: {best_score:.2f}")

            if best_score > 0.3: #nah kalau cosine similarytnya lebih dari 0.3, dianggap cukup mirip.
                response = df.iloc[best_idx]["answer"]
                response += "\n\nJika masalahmu belum teratasi, silakan hubungi admin."
                return response
            else:
                return "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."
        else:
            return "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."

# CLI Mode
if __name__ == "__main__":
    print("\n=== Chatbot Hybrid Test ===")
    while True:
        user_inp = input("Anda : ")
        if user_inp.lower() in ["exit", "quit", "keluar"]:
            print("Chatbot : Sampai jumpa!")
            break
        response = hybrid_chatbot(user_inp, model=default_model)
        print("Chatbot :", response)
