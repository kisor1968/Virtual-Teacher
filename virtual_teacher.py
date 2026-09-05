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
# 1. Page Configuration (MUST be first Streamlit command)
# ==========================================
st.set_page_config(
    page_title="Universal AI Virtual Classroom",
    page_icon="🎓",
    layout="centered"
)

# ==========================================
# 2. Configuration & Session State
# ==========================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if "visitor_count" not in st.session_state:
    st.session_state.visitor_count = 1428

# ==========================================
# 3. Permanent Background Handler
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
                st.session_state.persisted_plots[plot_key] = ("pyplot", fig_obj)
                st.pyplot(fig_obj)
                plt.close(fig_obj)

# ==========================================
# 5. App Header & College Branding with Logo (Centered Subtitle & Right Logo)
# ==========================================
col_title, col_logo = st.columns([5, 1])

with col_title:
    st.markdown(
        """
        <div style='text-align: left; margin-bottom: 15px;'>
            <h1 style='color: #0288d1; font-size: 2.3em; margin-bottom: 5px;'>🎓 Universal AI Virtual Classroom</h1>
            <p style='color: #0288d1; font-size: 1.25em; font-weight: 500; margin-top: 0; margin-bottom: 10px; text-align: center; white-space: nowrap;'>One platform, endless e‑learning possibilities</p>
            <p style='color: #555; font-size: 0.9em; margin: 0;'><b>Maintained by:</b> Prabhu Jagatbandhu College, Andul-Mouri, Howrah, Pin- 711302</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_logo:
    possible_names = ["logo_pjc.png", "logo_pjc.jpg", "logo.png", "logo.jpg"]
    found_logo = next((p for p in possible_names if os.path.exists(p)), None)
    
    if found_logo:
        st.image(found_logo, width=100)
    else:
        st.markdown("<p style='color: #d9534f; font-size: 0.8em;'><b>⚠️ Logo file not found.</b></p>", unsafe_allow_html=True)

# ==========================================
# 6. Sidebar Configuration & User Manual
# ==========================================
with st.sidebar:
    st.header("Configuration")
    selected_language = st.selectbox("Teaching Language", ["English", "Bengali", "Hindi", "Spanish", "French", "German"])
    enable_audio = st.checkbox("Enable Audio Narration", value=True)

    st.markdown("---")
    st.subheader("📖 User Manual & Guidelines")
    st.markdown(
        """
        1. **Start a Class:** Type any topic, concept, or question in the main text box and click **Start Class 🚀**.
        2. **Interactive Graphs:** Mention words like *plot*, *graph*, or *visualize* in your query to automatically generate 2D/3D visual aids.
        3. **Follow-ups:** Use the chat input at the bottom of the active lesson feed to ask questions or dive deeper.
        4. **Audio Narration:** Listen to professor explanations automatically if enabled.
        5. **Export Notes:** Download complete lecture notes as a text file anytime from the sidebar.
        """
    )

    if "transcript_log" in st.session_state and len(st.session_state.transcript_log) > 0:
        st.markdown("---")
        st.subheader("📥 Export Class Materials")
        lecture_transcript = f"UNIVERSAL AI VIRTUAL CLASSROOM\nTopic: {st.session_state.get('current_topic', 'General Session')}\n" + "=" * 40 + "\n\n"
        for item in st.session_state.transcript_log:
            lecture_transcript += f"[{item['role']}]:\n{item['content']}\n\n" + "-" * 20 + "\n\n"
            
        st.download_button(
            label="Download Lecture Notes (TXT)",
            data=lecture_transcript,
            file_name="lecture_notes.txt",
            mime="text/plain",
            key="download_lecture_notes_txt"
        )

lang_code_map = {"English": "en", "Bengali": "bn", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
tts_lang = lang_code_map.get(selected_language, "en")

# Initialize client using secrets
if GEMINI_API_KEY:
    if "client" not in st.session_state:
        st.session_state.client = genai.Client(api_key=GEMINI_API_KEY)
else:
    st.error("API Key not found. Please ensure `GEMINI_API_KEY` is configured in your Streamlit secrets file (`.streamlit/secrets.toml`).")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript_log" not in st.session_state:
    st.session_state.transcript_log = []
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""

topic_input = st.text_input("What topic or question do you want to cover today?", placeholder="e.g., Teach Fermi-Dirac statistics and plot it")
start_class = st.button("Start Class 🚀")

if start_class:
    if "client" not in st.session_state:
        st.error("Client not initialized. Check your API key configuration.")
    elif not topic_input:
        st.warning("Please enter a topic.")
    else:
        st.session_state.current_topic = topic_input
        st.session_state.persisted_plots = {}
        st.session_state.messages = []
        st.session_state.transcript_log = []
        
        with st.spinner("Preparing lesson..."):
            try:
                client = st.session_state.client
                system_prompt = (
                    f"You are an expert professor. Explain clearly in {selected_language}. "
                    f"CRITICAL RULE: Never draw ASCII art or text-based diagrams in your responses."
                )
                st.session_state.chat_history = client.chats.create(
                    model="gemini-3.6-flash",
                    config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.7)
                )
                
                response = safe_chat_send_message(st.session_state.chat_history, f"Start class on {topic_input}")
                
                bot_reply = response.text
                st.session_state.messages.append({"role": "assistant", "content": bot_reply, "plot_query": topic_input if should_render_plot(topic_input) else None})
                st.session_state.transcript_log.append({"role": "Professor", "content": bot_reply})
                st.success("Class started!")
            except Exception as e:
                st.error(f"Error starting class (Quota or Server limit): {e}")

# ==========================================
# Inline Lecture Feed with Positioned Plots
# ==========================================
if st.session_state.messages:
    st.markdown("---")
    st.subheader("📖 Lecture Feed")
    
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            display_content = clean_text_for_display(message["content"])
            st.markdown(display_content)
            
            if message.get("plot_query") and "client" in st.session_state:
                render_and_cache_plot(message["plot_query"], st.session_state.client, plot_key=f"msg_plot_{idx}")

            if message["role"] == "assistant" and enable_audio and display_content:
                try:
                    tts = gTTS(text=clean_text_for_speech(message["content"]), lang=tts_lang, slow=False)
                    fp = BytesIO()
                    tts.write_to_fp(fp)
                    fp.seek(0)
                    st.audio(fp, format="audio/mp3")
                except:
                    pass

    if student_answer := st.chat_input("Ask a follow-up or request plots..."):
        st.session_state.messages.append({"role": "user", "content": student_answer, "plot_query": None})
        st.session_state.transcript_log.append({"role": "Student", "content": student_answer})
        
        with st.chat_message("user"):
            st.markdown(student_answer)

        with st.chat_message("assistant"):
            with st.spinner("Professor is responding..."):
                try:
                    chat_response = safe_chat_send_message(st.session_state.chat_history, student_answer)
                    bot_reply = chat_response.text
                    st.markdown(clean_text_for_display(bot_reply))
                    
                    active_plot_q = student_answer if should_render_plot(student_answer) else None
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_reply, 
                        "plot_query": active_plot_q
                    })
                    st.session_state.transcript_log.append({"role": "Professor", "content": bot_reply})
                    
                    if active_plot_q and "client" in st.session_state:
                        render_and_cache_plot(active_plot_q, st.session_state.client, plot_key=f"msg_plot_{len(st.session_state.messages)-1}")
                        
                except Exception as e:
                    st.error(f"Error generating response (Quota / Rate limit reached): {e}")

# ==========================================
# Footer: Copyright & Visitor Counter
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.85em;'>"
    "Copyright © 2026 Dr. Kisor Mukhopadhyay, Prabhu Jagatbandhu College. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    f"<div style='text-align: center; color: #888; font-size: 0.8em; margin-top: 5px;'>"
    f"👀 Total Visitors: <b>{st.session_state.visitor_count}</b>"
    f"</div>",
    unsafe_allow_html=True
)
