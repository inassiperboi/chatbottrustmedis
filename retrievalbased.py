import pandas as pd
import re
import string

df = pd.read_csv("troubleshooting_dataset.csv")

# Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = text.strip()
    return text

df["clean_question"] = df["question"].apply(clean_text)
df["clean_answer"] = df["answer"].apply(clean_text)

# ======================
# 3. Encode Label Intent
# ======================
from sklearn.preprocessing import LabelEncoder

intent_encoder = LabelEncoder()
df["intent_id"] = intent_encoder.fit_transform(df["intent"])

# ======================
# 4. Split Train/Test
# ======================
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=42)

# ======================
# 5. TF-IDF Vectorizer
# ======================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train["clean_question"])
X_test = vectorizer.transform(test["clean_question"])

# ======================
# 6. Train Model
# ======================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["intent_id"])

# ======================
# 7. Evaluasi
# ======================
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(test["intent_id"], y_pred, target_names=intent_encoder.classes_))

# ======================
# 8. Chatbot Function
# ======================
def chatbot_response(user_input):
    # preprocessing
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    
    # prediksi intent
    pred_intent = model.predict(vectorized)[0]
    intent = intent_encoder.inverse_transform([pred_intent])[0]
    
    # ambil jawaban dari dataset sesuai intent
    answers = df[df["intent"] == intent]["answer"].unique()
    
    return {
        "intent": intent,
        "jawaban": answers[0] if len(answers) > 0 else "Maaf, saya belum punya jawaban untuk masalah itu."
    }

# ======================
# 9. Testing Chatbot
# ======================
while True:
    user_input = input("Anda: ")
    if user_input.lower() in ["exit", "quit", "keluar"]:
        print("Chatbot: Terima kasih, semoga membantu!")
        break
    
    response = chatbot_response(user_input)
    print(f"Chatbot (Intent: {response['intent']}): {response['jawaban']}")
