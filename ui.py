import streamlit as st
import pandas as pd
import numpy as np
import re, string, time
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Chatbot RS", page_icon="💬", layout="centered")

CONFIDENCE_THRESHOLD = 0.65
USE_FALLBACK = True

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

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    text = normalize_text(text)
    return text

@st.cache_resource
def load_and_train():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",      
        password="",       
        database="admisi_db"
    )
    query = "SELECT intent, question, answer, label FROM qa_dataset"
    df = pd.read_sql(query, conn)
    conn.close()

    # Preprocessing
    df["clean_question"] = df["question"].apply(clean_text)

    intent_encoder = LabelEncoder()
    df["intent_id"] = intent_encoder.fit_transform(df["intent"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_question"], df["intent_id"],
        test_size=0.2, random_state=42, stratify=df["intent_id"]
    )

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    return df, model, vectorizer, intent_encoder

df, model, vectorizer, intent_encoder = load_and_train()


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
            response = "Berikut beberapa solusi yang bisa dicoba:\n\n"
            for i, ans in enumerate(answers, 1):
                response += f"**Solusi {i}:** {ans}\n\n"
            response += "_Jika masalahmu belum teratasi, silakan hubungi admin._"
            return response
        else:
            return f"Maaf, belum ada jawaban untuk intent '{intent}'."
    else:
        if USE_FALLBACK:
            all_vectors = vectorizer.transform(df["clean_question"].tolist())
            sim_scores = cosine_similarity(vectorized, all_vectors).flatten()
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]

            if best_score > 0.3:
                return df.iloc[best_idx]["answer"] + "\n\n_Jika masalahmu belum teratasi, silakan hubungi admin._"
            else:
                return "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."
        else:
            return "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."

#ui
st.title("Trustmedis Chatbot")
# Avatar Chatbot (foto profil dari LinkedIn)
BOT_AVATAR = "https://media.licdn.com/dms/image/C5103AQEx__mbGbP-XA/profile-displayphoto-shrink_800_800/0/1583903115498?e=2147483647&v=beta&t=-ZcBVJc2wVULLx6ubiiyLdnYpmnXooYJGfXWxGeFjvA"

# Pesan pembuka
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Halo, saya **TrustBot**, asisten virtual RS. Ada yang bisa saya bantu hari ini?"}
    ]

# Render riwayat chat
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown(msg["content"])
    elif msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

# Input user
if prompt := st.chat_input("Ketik pertanyaanmu di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Jawaban bot
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Chatbot sedang mengetik..."):
            time.sleep(1.2)
            response = chatbot_response(prompt)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
