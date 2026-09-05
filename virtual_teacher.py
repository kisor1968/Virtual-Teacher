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
    text = re.sub(r"```[\s\S]*?
