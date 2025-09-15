import streamlit as st
import pandas as pd
import numpy as np
import re, string, time, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# =========================
# Konfigurasi
# =========================
CONFIDENCE_THRESHOLD = 0.65
USE_FALLBACK = True

# =========================
# Load & Preprocess Dataset
# =========================
df = pd.read_csv("troubleshooting_dataset.csv")

def normalize_text(text):
    replacements = {
        "satu sehat": "satusehat",
        "satu-sehat": "satusehat",
        "satu  sehat": "satusehat",
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

df["clean_question"] = df["question"].apply(clean_text)

# Encode intents
from sklearn.preprocessing import LabelEncoder
intent_encoder = LabelEncoder()
df["intent_id"] = intent_encoder.fit_transform(df["intent"])

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_question"], df["intent_id"], test_size=0.2, random_state=42, stratify=df["intent_id"]
)

# TF-IDF + Train model
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# =========================
# Chatbot Function
# =========================
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
            # Format response dengan HTML untuk dropdown
            response = "<div style='margin-bottom: 10px;'>Berikut beberapa solusi yang bisa dicoba:</div>"
            for i, answer in enumerate(answers, 1):
                response += f"<details style='margin-bottom: 8px;'><summary>Solusi {i}</summary><div style='padding: 8px; background: #f8f9fa; border-radius: 4px; margin-top: 5px;'>{answer}</div></details>"
            response += "<div style='margin-top: 15px; font-style: italic;'>Jika masalahmu belum teratasi, silakan hubungi admin.</div>"
            return response
        else:
            return f"Maaf, belum ada jawaban untuk intent '{intent}'. Silakan hubungi admin."
    else:
        if USE_FALLBACK:
            all_vectors = vectorizer.transform(df["clean_question"].tolist())
            sim_scores = cosine_similarity(vectorized, all_vectors).flatten()
            best_idx = np.argmax(sim_scores)
            best_score = sim_scores[best_idx]

            if best_score > 0.3:
                response = f"<div style='margin-bottom: 10px;'>{df.iloc[best_idx]['answer']}</div>"
                response += "<div style='font-style: italic;'>Jika masalahmu belum teratasi, silakan hubungi admin.</div>"
                return response
            else:
                return "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."
        else:
            return "Maaf, saya belum punya jawaban untuk masalah itu. Silakan hubungi admin."

# =========================
# UI with Streamlit
# =========================
st.set_page_config(page_title="Chatbot RS", page_icon="💬", layout="centered")

# Custom CSS untuk menghilangkan spasi yang tidak diinginkan
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stChatInput {
        bottom: 20px;
        position: fixed;
        width: 80%;
        left: 10%;
        background-color: white;
        z-index: 999;
        padding: 10px;
        border-radius: 20px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    .chat-container {
        padding-bottom: 80px;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 12px 16px;
        border-radius: 18px 18px 0px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .bot-message {
        background-color: #f0f0f0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        word-wrap: break-word;
    }
    .message-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .message-time {
        font-size: 0.7em;
        color: #888;
        margin-top: 8px;
        text-align: right;
    }
    .typing-indicator {
        background-color: #f0f0f0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0px;
        margin: 8px 0;
        max-width: 85%;
        margin-right: auto;
        font-style: italic;
        color: #888;
    }
    /* Styling untuk dropdown details */
    details {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        background: white;
    }
    summary {
        font-weight: bold;
        cursor: pointer;
        padding: 5px;
    }
    details[open] summary {
        margin-bottom: 8px;
        border-bottom: 1px solid #eee;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Trustmedis Chatbot")

BOT_AVATAR = "https://contacts.zoho.com/file?ot=8&t=serviceorg&ID=760599143"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("Chatbot", "👋 Halo! Ada yang bisa saya bantu terkait kendala di sistem Trustmedis?", datetime.datetime.now().strftime("%H:%M"))
    ]

# Container untuk chat
chat_container = st.container()

with chat_container:
    st.markdown('<div class="message-container">', unsafe_allow_html=True)
    
    for sender, message, timestamp in st.session_state.chat_history:
        if sender == "Anda":
            st.markdown(
                f"""
                <div class="user-message">
                    {message}
                    <div class="message-time">{timestamp}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif sender == "Chatbot":
            st.markdown(
                f"""
                <div style="display: flex; align-items: start; gap: 10px;">
                    <img src="{BOT_AVATAR}" width="36" style="border-radius: 50%;">
                    <div class="bot-message">
                        {message}
                        <div class="message-time">{timestamp}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif sender == "Typing":
            st.markdown(
                f"""
                <div style="display: flex; align-items: start; gap: 10px;">
                    <img src="{BOT_AVATAR}" width="36" style="border-radius: 50%;">
                    <div class="typing-indicator">
                        {message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

user_input = st.chat_input("Ketik pertanyaanmu di sini...")

if user_input:
    # 1. simpan pertanyaan user
    st.session_state.chat_history.append(("Anda", user_input, datetime.datetime.now().strftime("%H:%M")))
    # 2. tambahkan placeholder typing bubble
    st.session_state.chat_history.append(("Typing", "Chatbot sedang mengetik...", datetime.datetime.now().strftime("%H:%M")))
    st.session_state.last_user_input = user_input
    st.rerun()

# jika ada typing bubble, proses jawabannya
if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1][0] == "Typing":
    time.sleep(1.5)  # delay animasi
    user_input = st.session_state.last_user_input
    bot_response = chatbot_response(user_input)
    # hapus bubble typing
    st.session_state.chat_history.pop()
    # tambah jawaban bot
    st.session_state.chat_history.append(("Chatbot", bot_response, datetime.datetime.now().strftime("%H:%M")))
    st.rerun()