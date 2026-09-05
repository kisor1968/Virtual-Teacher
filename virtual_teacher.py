import os
import re
import time
import json
import base64
import streamlit as st
from google import genai
from google.genai import types
from gtts import gTTS
from io import BytesIO
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

# ==========================================
# 1. Page Configuration (MUST be first Streamlit command)
# ==========================================
st.set_page_config(
    page_title="Universal AI Virtual Classroom",
    page_icon="🎓",
    layout="centered"
)

# ==========================================
# 2. Configuration & Permanent Storage Setup
# ==========================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

FEEDBACK_FILE = "feedback_database.json"
VISITOR_FILE = "visitor_counter.json"

# Dynamic Visitor Counter Logic
if "visited" not in st.session_state:
    st.session_state.visited = True
    if os.path.exists(VISITOR_FILE):
        try:
            with open(VISITOR_FILE, "r") as f:
                data = json.load(f)
                visitor_count = data.get("count", 1428) + 1
        except Exception:
            visitor_count = 1429
    else:
        visitor_count = 1429
    
    with open(VISITOR_FILE, "w") as f:
        json.dump({"count": visitor_count}, f)
else:
    if os.path.exists(VISITOR_FILE):
        try:
            with open(VISITOR_FILE, "r") as f:
                visitor_count = json.load(f).get("count", 1428)
        except Exception:
            visitor_count = 1428
    else:
        visitor_count = 1428

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_feedback(rating, comment):
    reviews = load_feedback()
    new_review = {
        "rating": rating,
        "comment": comment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    reviews.append(new_review)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(reviews, f, indent=4)
    return reviews

# ==========================================
# 3. Permanent Background Handler & Custom Input Label Styling
# ==========================================
BG_IMAGE_PATH = "my_background.png"

bg_css = ""
if os.path.exists(BG_IMAGE_PATH):
    with open(BG_IMAGE_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    bg_css = f"""
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), url(data:image/png;base64,{encoded_image});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    """

custom_css = f"""
<style>
{bg_css}
.main .block-container {{
    background-color: rgba(255, 255, 255, 0.92);
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}
div[data-baseweb="input"] input, .stTextInput label p, .stTextArea label p {{
    color: #0288d1 !important;
    font-weight: 600;
}}
.stMarkdown, p, span, li {{
    color: #000000;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 4. Helper Functions
# ==========================================
def clean_text_for_speech(text):
    bt3 = chr(96) * 3
    bt1 = chr(96)
    text = re.sub(r'<script.*?>[\s\S]*?</script>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<script.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(bt3 + r'[\s\S]*?' + bt3, '', text)
    text = re.sub(bt1 + r'(.*?)' + bt1, r'\1', text)
    text = text.replace('$', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_for_display(text):
    text = re.sub(r'<script.*?>[\s\S]*?</script>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<script.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'---\s*###', r'\n\n###', text)
    return text.strip()

def should_render_plot(query):
    plot_keywords = ["plot", "graph", "visualize", "draw", "curve", "distribution", "surface", "chart"]
    return any(keyword in query.lower() for keyword in plot_keywords)

def safe_generate_content(client, model_name, contents, config=None):
    max_retries = 4
    delay = 3
    for attempt in range(max_retries):
        try:
            if config:
                return client.models.generate_content(model=model_name, contents=contents, config=config)
            else:
                return client.models.generate_content(model=model_name, contents=contents)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise e
    raise Exception("API quota limit reached or server busy. Please wait a minute and try again.")

def safe_chat_send_message(chat_session, message_text):
    max_retries = 4
    delay = 3
    for attempt in range(max_retries):
        try:
            return chat_session.send_message(message_text)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "503" in err_str:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise e
    raise Exception("API quota limit reached or server busy. Please wait a minute and try again.")

def generate_quiz_content(topic, client, language):
    prompt = (
        f"Generate a 3-question multiple-choice quiz in {language} on the topic: '{topic}'. "
        "Return the output STRICTLY as a valid JSON array of objects. Each object must have: "
        "'question' (string), 'options' (list of 4 strings), and 'answer' (the exact correct string from the options). "
        "Do not include any extra commentary or text outside the JSON array."
    )
    try:
        response = safe_generate_content(client, 'gemini-3.6-flash', prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("
