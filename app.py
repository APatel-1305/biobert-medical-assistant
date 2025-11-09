#app.py
import re
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification

# =====================================================
# 🩹 Patch model loader to fix safetensor meta issue
# =====================================================
_original_from_pretrained = AutoModelForSequenceClassification.from_pretrained
def safe_from_pretrained(model_dir, *args, **kwargs):
    kwargs.setdefault("torch_dtype", torch.float32)
    kwargs.setdefault("low_cpu_mem_usage", False)
    return _original_from_pretrained(model_dir, *args, **kwargs)
AutoModelForSequenceClassification.from_pretrained = safe_from_pretrained

# =====================================================
# Import assistant logic
# =====================================================
from assistant import reply

# =====================================================
# Streamlit Page Config
# =====================================================
st.set_page_config(
    page_title="🩺 BioBERT Medical Assistant",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for chat style
st.markdown("""
    <style>
        body { background-color: #f8f9fa; }
        .user-bubble {
            background-color: #d0e3ff;
            color: #000;
            padding: 12px 16px;
            border-radius: 20px;
            margin-bottom: 10px;
            max-width: 80%;
            align-self: flex-end;
        }
        .bot-bubble {
            background-color: #e9ecef;
            color: #000;
            padding: 12px 16px;
            border-radius: 20px;
            margin-bottom: 10px;
            max-width: 80%;
            align-self: flex-start;
        }
        .disclaimer-inline {
            display: block;
            margin-top: 8px;
            font-size: 0.85em;
            color: #555;
            font-weight: 600;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .disclaimer {
            font-size: 0.9em;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            margin-top: 15px;
            padding-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# Header and Info
# =====================================================
st.title("🩺 BioBERT Medical Dialogue Assistant")

st.markdown("""
This assistant uses **BioBERT** for medical response ranking and **FAISS** for fast information retrieval.  
It provides *informational* responses — not professional medical advice.  
""")

with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    **Powered by:**  
    - 🧠 BioBERT (`dmis-lab/biobert-base-cased-v1.1`)  
    - 🧩 FAISS Retrieval  
    - 💬 Sentence Transformers (`all-MiniLM-L6-v2`)
    """)
    st.markdown("""
    **Folders:**  
    - `biobert_ranker/` → Trained model  
    - `faiss_responses.index` → Vector index  
    """)
    if st.button("🧹 Clear Conversation"):
        st.session_state.history = []
        st.rerun()

# =====================================================
# Initialize Session State
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# Main Input Section
# =====================================================
query = st.text_input("👤 Enter your medical question:", placeholder="Describe your symptoms or ask a question...")

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("Get Doctor Response", use_container_width=True):
        if query.strip():
            with st.spinner("🧠 Retrieving and ranking responses..."):
                try:
                    answer = reply(query)

                    # 🧹 Remove “SOURCE: ...” and confidence info if present
                    clean_answer = re.sub(r"\(SOURCE:.*?\)", "", answer, flags=re.IGNORECASE)
                    clean_answer = re.sub(r"confidence\s*\d+(\.\d+)?", "", clean_answer, flags=re.IGNORECASE)
                    clean_answer = re.sub(r"\n{2,}", "\n\n", clean_answer).strip()

                    st.session_state.history.append((query, clean_answer))
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# =====================================================
# Display Chat History
# =====================================================
st.markdown("### 💬 Conversation")

chat_html = '<div class="chat-container">'
for patient_msg, bot_msg in st.session_state.history:
    chat_html += f'<div class="user-bubble">👤 <b>Patient:</b> {patient_msg}</div>'

    # 🩹 Format the assistant message — bold + small disclaimer with line gap
    formatted_msg = re.sub(
        r"(DISCLAIMER[:\-].*)",
        r'<br><span class="disclaimer-inline"><b>\1</b></span>',
        bot_msg,
        flags=re.IGNORECASE
    )

    chat_html += f'<div class="bot-bubble">🤖 <b>Assistant:</b> {formatted_msg}</div>'
chat_html += "</div>"

st.markdown(chat_html, unsafe_allow_html=True)

# =====================================================
# Footer Disclaimer
# =====================================================
st.markdown("""
<div class="disclaimer">
⚠️ <b>Disclaimer:</b> The responses generated are for **educational and informational** purposes only.  
They do not constitute medical advice. Always consult a qualified healthcare professional for serious health issues.
</div>
""", unsafe_allow_html=True)
