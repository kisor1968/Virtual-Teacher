import os
import re
import time
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
# CONFIGURATION: Securely loaded via Streamlit Secrets
# ==========================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Page Configuration
st.set_page_config(
    page_title="Universal AI Virtual Classroom",
    page_icon="🎓",
    layout="centered"
)

# Initialize Visitor Counter in Session State
if "visitor_count" not in st.session_state:
    st.session_state.visitor_count = 1428

# ==========================================
# Permanent Background Handler
# ==========================================
BG_IMAGE_PATH = "my_background.png"

if os.path.exists(BG_IMAGE_PATH):
    with open(BG_IMAGE_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    custom_css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), url(data:image/png;base64,{encoded_image});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# Helper functions
def clean_text_for_speech(text):
    text = re.sub(r'<script.*?>[\s\S]*?</script>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<script.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r"```[\s\S]*?```", '', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
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

# ==========================================
# Robust LLM Content Generation with Quota Management
# ==========================================
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

# ==========================================
# Universal Dynamic AI Plotter Engine (Context-True)
# ==========================================
def render_and_cache_plot(user_query, client, plot_key):
    if "persisted_plots" not in st.session_state:
        st.session_state.persisted_plots = {}

    if plot_key in st.session_state.persisted_plots:
        cached_type, cached_data = st.session_state.persisted_plots[plot_key]
        if cached_type == "plotly":
            st.plotly_chart(cached_data, use_container_width=True)
        else:
            st.pyplot(cached_data)
        return

    query_lower = user_query.lower()
    is_3d = any(k in query_lower for k in ["3d", "surface", "three-dimensional", "three dimensional", "z("])
    
    safe_globals = {
        "np": np, "plt": plt, "go": go, "sp": sp,
        "k_B": 1.380649e-23, "kB": 1.380649e-23, 
        "R": 8.314, "h": 6.62607015e-34, 
        "c": 299792458, "N_A": 6.02214076e23
    }
    
    if is_3d:
        with st.status("Generating 3D Surface Graph...", expanded=True):
            code_text = None
            try:
                prompt = (
                    f"Write a Python script using numpy (as np) and plotly.graph_objects (as go) "
                    f"to create an interactive 3D surface plot precisely matching: '{user_query}'. "
                    "Return ONLY executable Python code inside a markdown code block. "
                    "Assign the final Plotly figure to a variable named `fig`."
                )
                response = safe_generate_content(client, 'gemini-3.6-flash', prompt)
                code_text = response.text
            except Exception:
                pass
            
            try:
                code_match = re.search(r"```python\s*(.*?)\s*
