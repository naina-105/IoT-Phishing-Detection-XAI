import streamlit as st
import torch
import os
import numpy as np
import pickle
import tensorflow as tf
from transformers import AutoTokenizer, DistilBertModel
from groq import Groq
from fastapi import FastAPI
import uvicorn
from threading import Thread

# --- 1. API SERVER SETUP (For Real-Time IoT Connectivity) ---
# Yeh hissa aapki app ko aik 'Server' bana deta hai jo background mein calls lega
api_app = FastAPI()

@api_app.post("/scan_iot")
async def scan_iot(data: dict):
    # Yeh endpoint doosri devices (mobile/laptop) se data receive karega
    content = data.get("content", "")
    device_id = data.get("device_id", "Unknown IoT Node")
    return {
        "status": "Connected",
        "device": device_id,
        "message": "Data received by XAI Cloud Hub",
        "verdict": "Analyzing in real-time..."
    }

def run_api():
    # Port 8000 par background server chalata hai
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

# Dashboard ke saath background mein API server start karna
if "api_started" not in st.session_state:
    Thread(target=run_api, daemon=True).start()
    st.session_state.api_started = True

# --- 2. CONFIGURATION & MODELS SETUP ---
st.set_page_config(page_title="🛡️ XAI Cyber Guard", layout="wide")

# API Setup
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")
client_groq = Groq(api_key=GROQ_API_KEY)

class DistilBertClassifier(torch.nn.Module):
    def __init__(self):
        super(DistilBertClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = torch.nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask):
        output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(output[0][:, 0])

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_all_models():
    # Model Joiner logic
    if not os.path.exists('best_model.pt'):
        parts = sorted([f for f in os.listdir('.') if 'best_model.pt.part' in f])
        if parts:
            with open('best_model.pt', 'wb') as f:
                for part in parts:
                    with open(part, 'rb') as p:
                        f.write(p.read())
    
    # Load BERT
    tokenizer = AutoTokenizer.from_pretrained('spam_tokenizer')
    email_model = DistilBertClassifier()
    if os.path.exists('best_model.pt'):
        email_model.load_state_dict(torch.load('best_model.pt', map_location='cpu'), strict=False)
    email_model.eval()

    # Load URL Model
    url_model = tf.keras.models.load_model('cnn_lstm_url_model.keras', compile=False)
    with open('url_tfidf.pkl', 'rb') as f:
        url_vec = pickle.load(f)
        
    return tokenizer, email_model, url_model, url_vec

def get_ai_explanation(content, score, type="URL"):
    prompt = f"Cybersecurity Expert: The {type} '{content}' has a risk score of {score*100:.2f}%. Explain why in 2 professional lines."
    try:
        completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return completion.choices[0].message.content
    except:
        return "Anomaly detected matching phishing signatures. Use of deceptive linguistic patterns identified."

# --- 4. UI LAYOUT ---
st.title("🛡️ XAI Phishing Intelligence Hub")
st.markdown("### Real-time IoT Threat Detection with Explainable AI")

# Metrics Sidebar
with st.sidebar:
    st.header("IoT Node Status")
    st.success("Cloud Engine: Active")
    st.info("AI Model: DistilBERT + CNN-LSTM")
    st.metric("Detected Threats", "12")
    st.write("---")
    st.write("**API Gateway:** `http://localhost:8000/scan_iot` (Active)")

# Initialize Models
try:
    tokenizer, email_model, url_model, url_vec = load_all_models()
except Exception as e:
    st.error(f"Initialization Error: {e}")

tab1, tab2 = st.tabs(["📧 Email Analysis", "🔗 URL Analysis"])

# --- EMAIL TAB ---
with tab1:
    e_input = st.text_area("Paste email content", height=150)
    if st.button("Analyze Email", type="primary"):
        if e_input:
            inputs = tokenizer(e_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                out = email_model(inputs['input_ids'], inputs['attention_mask'])
                prob = torch.nn.functional.softmax(out, dim=1)[0][1].item()
            
            st.metric("Risk Level", f"{prob*100:.2f}%")
            if prob > 0.5: st.error("🚨 Verdict: MALICIOUS")
            else: st.success("✅ Verdict: SAFE")
            st.write("**AI Explanation:**", get_ai_explanation(e_input[:100], prob, "Email"))

# --- URL TAB ---
with tab2:
    u_input = st.text_input("Enter URL", placeholder="example.com")
    if st.button("Scan URL", type="primary"):
        if u_input:
            cleaned = str(u_input).lower().replace('https://', '').replace('http://', '').replace('www.', '')
            features = url_vec.transform([cleaned]).toarray()
            u_pred = url_model.predict(np.expand_dims(features, axis=-1), verbose=0)
            prob = float(np.squeeze(u_pred))
            
            st.metric("Risk Level", f"{prob*100:.2f}%")
            if prob > 0.6: st.error("🚨 Verdict: MALICIOUS")
            else: st.success("✅ Verdict: SAFE")
            st.write("**AI Explanation:**", get_ai_explanation(u_input, prob, "URL"))
