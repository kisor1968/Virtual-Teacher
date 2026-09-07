import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import google.generativeai as genai

# Page Configuration
st.set_page_config(
    page_title="Virtual AI Classroom",
    page_icon="🧑‍🏫",
    layout="wide"
)

# Custom CSS for Background Image & Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("YOUR_BACKGROUND_IMAGE_URL_HERE");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_topic" not in st.session_state:
    st.session_state["current_topic"] = "General Studies"

# Auto-configure API key from Streamlit Secrets if available
if "client" not in st.session_state:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            st.session_state["client"] = genai
    except Exception:
        pass

# ==========================================
# Sidebar Configuration & RAG Materials
# ==========================================
st.sidebar.title("📚 Classroom Controls")

# Fallback manual API key input if secrets aren't set
if "client" not in st.session_state:
    api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state["client"] = genai
            st.sidebar.success("API Key configured successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error configuring API key: {e}")
else:
    st.sidebar.success("✅ AI Connected via Secrets")

st.sidebar.markdown("---")
selected_language = st.sidebar.selectbox(
    "Select Language / Medium", 
    ["English", "Spanish", "French", "German", "Hindi", "Bengali"]
)

st.sidebar.markdown("### 📄 Study Materials (RAG)")
uploaded_files = st.sidebar.file_uploader(
    "Upload textbooks, PDFs, or notes", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)
