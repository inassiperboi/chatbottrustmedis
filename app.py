from flask import Flask, render_template, request, jsonify
import pandas as pd
import re, string, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Konfigurasi
CONFIDENCE_THRESHOLD = 0.65
USE_FALLBACK = True  

# Load Dataset
df = pd.read_csv("troubleshooting_dataset.csv")

# Normalisasi teks
def normalize_text(text):
    replacements = {
        "satu sehat": "satusehat",
        "satu-sehat": "satusehat",
        "bpjs kesehatan": "bpjs",
        "rekam medis": "rm"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    text = normalize_text(text)
    return text

df["clean_question"] = df["question"].apply(clean_text)

# Encode intents
intent_encoder = LabelEncoder()
df["intent_id"] = intent_encoder.fit_transform(df["intent"])

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_question"], df["intent_id"], test_size=0.2, random_state=42, stratify=df["intent_id"]
)

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Chatbot logic
def chatbot_response(user_input):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    pred_proba = model.predict_proba(vectorized)[0]
    confidence = max(pred_proba)
    pred_intent = model.predict(vectorized)[0]
    intent = intent_encoder.inverse_transform([pred_intent])[0]

    if confidence >= CONFIDENCE_THRESHOLD:
        answers = df[df["intent"] == intent]["answer"].unique().tolist()
        if answers:
            response = "Berikut beberapa solusi yang bisa dicoba:<br>• " + "<br>• ".join(answers)
            response += "<br><br>Jika masalahmu belum teratasi, silakan hubungi admin."
        else:
            response = "Maaf, belum ada jawaban untuk intent ini. Silakan hubungi admin."
    else:
        if USE_FALLBACK:
            all_vectors = vectorizer.transform(df["clean_question"].tolist())
            sim_scores = cosine_similarity(vectorized, all_vectors).flatten()
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]

            if best_score > 0.3:
                response = df.iloc[best_idx]["answer"]
                response += "<br><br>Jika masalahmu belum teratasi, silakan hubungi admin."
            else:
                response = "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."
        else:
            response = "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."

    return response

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json["message"]
    bot_reply = chatbot_response(user_input)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
