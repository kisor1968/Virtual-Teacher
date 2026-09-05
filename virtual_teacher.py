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

# Helper functions using dynamic chr(96) to prevent syntax conflicts
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
    
    bt3 = chr(96) * 3
    code_pattern = f"{bt3}python\\s*(.*?)\\s*{bt3}"

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
                code_match = re.search(code_pattern, code_text or "", re.DOTALL)
                if code_match:
                    exec_code = code_match.group(1)
                    local_vars = {}
                    exec(exec_code, safe_globals, local_vars)
                    fig = local_vars.get("fig") or safe_globals.get("fig")
                    if fig:
                        st.session_state.persisted_plots[plot_key] = ("plotly", fig)
                        st.plotly_chart(fig, use_container_width=True)
                        return
            except Exception:
                pass
                
            x = np.linspace(-5, 5, 40)
            y = np.linspace(-5, 5, 40)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y)
            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
            st.session_state.persisted_plots[plot_key] = ("plotly", fig)
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        with st.status("Generating Universal Graph...", expanded=True):
            code_text = None
            try:
                prompt = (
                    f"Write a Python script using numpy (as np), scipy.special (as sp), and matplotlib.pyplot (as plt) "
                    f"to create a clear 2D scientific plot with explicit axis labels, title, grid, and legend for the exact request: '{user_query}'. "
                    "Return ONLY executable Python code inside a markdown code block. "
                    "Use plt.figure(figsize=(8, 4), facecolor='white') and call plt.tight_layout(). Do NOT call plt.show()."
                )
                response = safe_generate_content(client, 'gemini-3.6-flash', prompt)
                code_text = response.text
            except Exception:
                pass

            try:
                code_match = re.search(code_pattern, code_text or "", re.DOTALL)
                plt.close('all')
                
                if code_match:
                    exec_code = code_match.group(1)
                    local_vars = {}
                    exec(exec_code, safe_globals, local_vars)
                
                if plt.get_fignums():
                    fig_obj = plt.gcf()
                    fig_obj.patch.set_facecolor('white')
                    plt.tight_layout()
                else:
                    fig_obj = plt.figure(figsize=(8, 4), facecolor='white')
                    x = np.linspace(-4, 4, 400)
                    if "fermi" in query_lower:
                        E_F = 0.5
                        f_FD = 1.0 / (np.exp((x - E_F) / 0.1) + 1.0)
                        plt.plot(x, f_FD, color='blue', lw=2, label="Fermi-Dirac")
                        plt.title("Fermi-Dirac Distribution")
                    elif "maxwell" in query_lower:
                        v = np.linspace(0, 2000, 400)
                        plt.plot(v, v**2 * np.exp(-v**2/500000), color='crimson', lw=2, label="Maxwell-Boltzmann")
                        plt.title("Maxwell-Boltzmann Speed Distribution")
                    else:
                        plt.plot(x, np.sin(x), color='darkorange', lw=2, label=user_query)
                        plt.title(f"Plot for: {user_query}")
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.legend(loc='best')
                    plt.tight_layout()

                st.session_state.persisted_plots[plot_key] = ("pyplot", fig_obj)
                st.pyplot(fig_obj)
                plt.close(fig_obj)
                
            except Exception as e:
                st.warning(f"Plotting fallback engaged: {e}")
                fig_obj = plt.figure(figsize=(8, 4), facecolor='white')
                x = np.linspace(-4, 4, 400)
                plt.plot(x, np.sin(x), color='purple', lw=2, label="Fallback Plot")
                plt.title(f"Graph: {user_query}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()
                st.session_state.persisted_plots[plot_key] = ("pyplot
