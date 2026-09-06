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
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

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
div[data-baseweb="input"] input, .stTextInput label p, .stTextArea label p, .stFileUploader label p, .stAudioInput label p {{
    color: #0288d1 !important;
    font-weight: 600;
}}
.stMarkdown, p, span, li {{
    color: #000000;
}}
.benefits-box {{
    background: #e1f5fe;
    border-left: 5px solid #0288d1;
    padding: 15px 20px;
    border-radius: 6px;
    margin-bottom: 20px;
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

def extract_json_block(text):
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            subparts = parts[1].split("```")
            if len(subparts) > 0:
                return subparts[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return text.strip()

def generate_quiz_content(topic, client, language):
    prompt = (
        f"Generate exactly 3 multiple-choice questions in {language} on the topic: '{topic}'. "
        "Return the output STRICTLY as a valid JSON array of 3 objects. Each object must have: "
        "'question' (string), 'options' (list of 4 strings), and 'answer' (the exact correct string from the options). "
        "Do not include any extra commentary or text outside the JSON array."
    )
    try:
        response = safe_generate_content(client, 'gemini-3.6-flash', prompt)
        text = extract_json_block(response.text)
            
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]
            
        parsed_data = json.loads(text)
        if isinstance(parsed_data, list) and len(parsed_data) >= 3:
            return parsed_data[:3]
    except Exception:
        pass
        
    return [
        {
            "question": f"What is a foundational principle of {topic}?",
            "options": ["Core fundamentals", "Unrelated concepts", "Arbitrary noise", "None of the above"],
            "answer": "Core fundamentals"
        },
        {
            "question": f"Which of the following best characterizes {topic}?",
            "options": ["Systematic analysis", "Random occurrence", "Static state", "Undefined behavior"],
            "answer": "Systematic analysis"
        },
        {
            "question": f"Why is understanding {topic} important?",
            "options": ["For advanced application", "It has no practical use", "Only for historical records", "It is entirely theoretical"],
            "answer": "For advanced application"
        }
    ]

def generate_homework_prompt(topic, client, language):
    prompt = (
        f"As an expert professor, create a thoughtful homework assignment task or problem for students learning about '{topic}' in {language}. "
        "Provide a clear, concise prompt or question that the student needs to answer."
    )
    try:
        response = safe_generate_content(client, 'gemini-3.6-flash', prompt)
        return response.text.strip()
    except Exception:
        return f"Explain the core concepts of {topic} and provide a real-world example of its application."

def generate_flashcards(transcript_text, client, language):
    prompt = (
        f"Based on the following lecture transcript, generate exactly 4 study flashcards in {language}. "
        "Return the output STRICTLY as a valid JSON array of 4 objects. Each object must have: "
        "'front' (string representing the question/term) and 'back' (string representing the answer/explanation). "
        "Do not include any extra commentary or text outside the JSON array.\n\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )
    try:
        response = safe_generate_content(client, 'gemini-3.6-flash', prompt)
        text = extract_json_block(response.text)
            
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx+1]
            
        parsed_data = json.loads(text)
        if isinstance(parsed_data, list) and len(parsed_data) >= 4:
            return parsed_data[:4]
    except Exception:
        pass
        
    return [
        {"front": "Core Concept 1", "back": "Primary foundational principle explained in the lesson."},
        {"front": "Core Concept 2", "back": "Key mechanism or methodology discussed."},
        {"front": "Practical Application", "back": "Real-world implementation of the lesson topic."},
        {"front": "Summary Takeaway", "back": "Crucial conclusion from the virtual classroom session."}
    ]

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
# 5. App Header & College Branding with Logo
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
# Front-Page Importance & Why Students Should Use It
# ==========================================
st.markdown(
    """
    <div class="benefits-box">
        <h4 style="color: #0288d1; margin-top: 0; margin-bottom: 8px;">🌟 Why Use This Virtual Classroom? (The AI Advantage)</h4>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.95em; line-height: 1.5; color: #333;">
            <b>Structured Learning vs. Chat:</b> Unlike standard AI chatbots that just output text paragraphs, this platform structures your education with interactive lectures, automated quizzes, and active tutor evaluations.
            <br><b>Dynamic Scientific Visualizations:</b> Automatically plots complex mathematical, physical, and statistical equations (2D & 3D) on demand.
            <br><b>Multimodal & Multi-Document RAG:</b> Ground your lessons on uploaded study materials (PDFs/TXT) and speak directly via voice input.
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 6. Sidebar Configuration & User Manual
# ==========================================
with st.sidebar:
    st.header("Configuration")
    selected_language = st.selectbox("Teaching Language", ["English", "Bengali", "Hindi", "Spanish", "French", "German"])
    enable_audio = st.checkbox("Enable Audio Narration", value=True)

    st.markdown("---")
    st.subheader("📚 Multi-Document RAG Materials")
    uploaded_rag_files = st.file_uploader(
        "Upload Study Materials (PDF, TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks, or research papers for the AI tutor to ground its teaching on."
    )

    st.markdown("---")
    st.subheader("📖 User Manual & Guidelines")
    st.markdown(
        """
        1. **Virtual Classroom:** Upload study materials (RAG) and enter a topic to launch an interactive lecture.
        2. **Voice-to-Voice AI Tutor:** Record your verbal questions directly using your microphone.
        3. **Knowledge Check:** Test understanding using generated 3-question lesson quizzes.
        4. **Assignment Evaluator:** Have the AI assign homework and evaluate your submissions.
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

    st.markdown("---")
    st.subheader("⭐ Feedback & Review")
    
    with st.form("sidebar_feedback_form"):
        star_rating = st.slider("Star Rating", 1, 5, 5, format="%d ⭐")
        comment_text = st.text_area("Your Comment", placeholder="Share your experience or suggestions...")
        submit_review = st.form_submit_button("Submit & Record Review")
        
        if submit_review:
            save_feedback(star_rating, comment_text)
            st.success("Review saved permanently!")

    all_reviews = load_feedback()
    if all_reviews:
        with st.expander(f"📋 View All Reviews ({len(all_reviews)})"):
            for idx, rev in enumerate(reversed(all_reviews), 1):
                stars = "⭐" * rev["rating"]
                st.markdown(f"**#{len(all_reviews) - idx + 1} - {stars}**")
                if rev["comment"]:
                    st.caption(f'"{rev["comment"]}"')
                st.text(f"🕒 {rev['timestamp']}")
                st.divider()

lang_code_map = {"English": "en", "Bengali": "bn", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
tts_lang = lang_code_map.get(selected_language, "en")

# Initialize client using secrets
if GEMINI_API_KEY:
    if "client" not in st.session_state:
        st.session_state.client = genai.Client(api_key=GEMINI_API_KEY)
else:
    st.error("API Key not found. Please ensure `GEMINI_API_KEY` is configured in your Streamlit secrets file (`.streamlit/secrets.toml`).")

# Session state initializations
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcript_log" not in st.session_state:
    st.session_state.transcript_log = []
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "homework_prompt" not in st.session_state:
    st.session_state.homework_prompt = ""
if "flashcards_data" not in st.session_state:
    st.session_state.flashcards_data = None

# ==========================================
# Main App Layout
# ==========================================
topic_input = st.text_input("What topic or question do you want to cover today?", placeholder="e.g., Teach Fermi-Dirac statistics based on the uploaded materials")
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
        st.session_state.quiz_data = None
        st.session_state.homework_prompt = ""
        st.session_state.flashcards_data = None
        
        with st.spinner("Preparing lesson and analyzing study materials (RAG)..."):
            try:
                client = st.session_state.client
                system_prompt = (
                    f"You are an expert professor. Explain clearly in {selected_language}. "
                    f"CRITICAL RULE: Never draw ASCII art or text-based diagrams in your responses. "
                    f"Incorporate insights and details from any provided study documents (RAG materials) when answering."
                )
                st.session_state.chat_history = client.chats.create(
                    model="gemini-3.6-flash",
                    config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.7)
                )
                
                initial_payload = []
                if uploaded_rag_files:
                    initial_payload.append("Here are the study reference documents for this lesson:")
                    for file in uploaded_rag_files:
                        file_bytes = file.read()
                        if file.type == "application/pdf":
                            initial_payload.append(types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"))
                        else:
                            try:
                                text_content = file_bytes.decode("utf-8")
                                initial_payload.append(f"Document Name: {file.name}\nContent:\n{text_content}")
                            except:
                                initial_payload.append(types.Part.from_bytes(data=file_bytes, mime_type="text/plain"))
                
                initial_payload.append(f"Start class on: {topic_input}")
                
                response = safe_chat_send_message(st.session_state.chat_history, initial_payload)
                
                bot_reply = response.text
                st.session_state.messages.append({"role": "assistant", "content": bot_reply, "plot_query": topic_input if should_render_plot(topic_input) else None})
                st.session_state.transcript_log.append({"role": "Professor", "content": bot_reply})
                st.success("Class started with multi-document RAG context!")
            except Exception as e:
                st.error(f"Error starting class (Quota or Server limit): {e}")

# Inline Lecture Feed with Positioned Plots
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

    # ==========================================
    # Voice-to-Voice AI Tutor Input Section
    # ==========================================
    st.markdown("---")
    st.subheader("🎙️ Voice-to-Voice AI Tutor (Speak to Ask)")
    st.markdown("Record a voice question using your microphone, and the AI professor will listen and respond.")
    
    spoken_audio = st.audio_input("Record your verbal question for the professor")
    
    if spoken_audio is not None and "client" in st.session_state:
        audio_bytes = spoken_audio.read()
        audio_key = hash(audio_bytes)
        
        if st.session_state.get("last_processed_audio") != audio_key:
            st.session_state.last_processed_audio = audio_key
            
            with st.spinner("Professor is listening to your voice..."):
                try:
                    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                    audio_prompt = [
                        audio_part,
                        f"Listen to the student's voice question regarding the current class topic '{st.session_state.current_topic}' and answer thoroughly as an expert professor in {selected_language}."
                    ]
                    
                    chat_response = safe_chat_send_message(st.session_state.chat_history, audio_prompt)
                    bot_reply = chat_response.text
                    
                    user_voice_label = "🎤 [Spoken Audio Question from Student]"
                    st.session_state.messages.append({"role": "user", "content": user_voice_label, "plot_query": None})
                    st.session_state.transcript_log.append({"role": "Student", "content": user_voice_label})
                    
                    active_plot_q = None
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_reply, 
                        "plot_query": active_plot_q
                    })
                    st.session_state.transcript_log.append({"role": "Professor", "content": bot_reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing voice input: {e}")

    # Standard Text Chat Input Alternative
    if student_answer := st.chat_input("Or type a follow-up / request plots..."):
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
    # Vertical Section 1: Automated Summary & Flashcard Generator
    # ==========================================
    st.markdown("---")
    st.subheader("💡 Lesson Flashcards & Summary Generator")
    st.markdown("Generate concise study flashcards extracted directly from today's lecture conversation transcript.")

    if st.button("Generate Study Flashcards"):
        with st.spinner("Analyzing transcript and creating flashcards..."):
            full_transcript_str = "\n".join([f"{item['role']}: {item['content']}" for item in st.session_state.transcript_log])
            st.session_state.flashcards_data = generate_flashcards(
                full_transcript_str,
                st.session_state.client,
                selected_language
            )

    if st.session_state.flashcards_data:
        st.markdown("#### 🃏 Interactive Flashcards (Click to Expand Answers)")
        for fc_idx, card in enumerate(st.session_state.flashcards_data):
            with st.expander(f"Card {fc_idx+1}: {card['front']}"):
                st.markdown(f"**Answer / Explanation:**\n{card['back']}")

    # ==========================================
    # Vertical Section 2: Knowledge Check & Quiz (Guaranteed 3 Questions)
    # ==========================================
    st.markdown("---")
    st.subheader("🧠 Knowledge Check & Quiz")
    st.markdown("Test your understanding based on what was just taught in today's class!")

    if st.button("Generate Quiz for this Lesson"):
        with st.spinner("Generating 3 quiz questions..."):
            st.session_state.quiz_data = generate_quiz_content(
                st.session_state.current_topic, 
                st.session_state.client, 
                selected_language
            )

    if st.session_state.quiz_data:
        with st.form("lesson_quiz_form"):
            user_selections = {}
            for q_idx, q_item in enumerate(st.session_state.quiz_data):
                st.markdown(f"**Q{q_idx+1}: {q_item['question']}**")
                user_selections[q_idx] = {}
                for opt_idx, option in enumerate(q_item['options']):
                    user_selections[q_idx][option] = st.checkbox(
                        option, 
                        value=False, 
                        key=f"quiz_q_{q_idx}_opt_{opt_idx}"
                    )
                st.write("")
            
            submit_quiz = st.form_submit_button("Submit Quiz Answers")
            
            if submit_quiz:
                score = 0
                st.markdown("### 📊 Quiz Results & Feedback")
                for q_idx, q_item in enumerate(st.session_state.quiz_data):
                    correct = q_item['answer']
                    chosen_options = [opt for opt, checked in user_selections[q_idx].items() if checked]
                    
                    if len(chosen_options) == 0:
                        st.warning(f"**Q{q_idx+1}:** No option selected. The correct answer was '{correct}'.")
                    elif len(chosen_options) > 1:
                        st.error(f"**Q{q_idx+1}:** Multiple options selected. Please select only one option per question. The correct answer was '{correct}'.")
                    else:
                        selected = chosen_options[0]
                        if selected == correct:
                            score += 1
                            st.success(f"**Q{q_idx+1}:** Correct! You chose '{selected}'.")
                        else:
                            st.error(f"**Q{q_idx+1}:** Incorrect. You chose '{selected}', but the correct answer is '{correct}'.")
                
                st.info(f"**Final Score: {score} / {len(st.session_state.quiz_data)}**")
# ==========================================
# Interactive Whiteboard Section
# ==========================================
st.markdown("---")
st.subheader("🎨 Interactive Whiteboard & Sketchpad")
st.markdown("Draw diagrams, geometric shapes, or handwritten equations for the AI professor to review.")

stroke_width = st.slider("Stroke Width", 1, 25, 3)
stroke_color = st.color_picker("Stroke Color", "#000000")
bg_color = st.color_picker("Background Color", "#ffffff")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=350,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Ask Professor to Analyze Drawing"):
    if canvas_result.image_data is not None and "client" in st.session_state:
        with st.spinner("Professor is analyzing your sketch..."):
            try:
                img_array = canvas_result.image_data.astype(np.uint8)
                success, encoded_img = cv2.imencode('.png', img_array)

                if success:
                    image_bytes = encoded_img.tobytes()
                    image_obj = Image.open(BytesIO(image_bytes))

                    drawing_prompt = [
                        image_obj,
                        f"Analyze this student's whiteboard drawing or handwritten formula regarding the topic '{st.session_state.get('current_topic', 'General Session')}'. "
                        f"Provide constructive academic feedback, correct any errors, and explain the concept clearly in {selected_language}."
                    ]

                    response = safe_generate_content(st.session_state.client, 'gemini-3.6-flash', drawing_prompt)

                    st.markdown("### 🧑‍🏫 Professor's Feedback on Your Sketch")
                    st.markdown(response.text)
                else:
                    st.error("Failed to process the canvas image.")
            except Exception as e:
                st.error(f"Error analyzing whiteboard sketch: {e}")
    else:
        st.warning("Please draw something on the canvas first or ensure your session is active.")
# ==========================================
# Vertical Section 3: Homework & Assignment Evaluator (Tutor-Assigned with File Upload)
# ==========================================
st.markdown("---")
st.subheader("📝 AI Homework & Assignment Evaluator")
st.markdown("The virtual tutor will assign a homework task based on today's lesson. Type your answer or upload a file (PDF, JPG, PNG) for evaluation.")

if st.button("Get Homework Task from Tutor"):
    if not st.session_state.current_topic:
        st.warning("Please start a class first so the tutor knows what topic to assign homework on.")
    else:
        with st.spinner("Tutor is preparing your assignment..."):
            st.session_state.homework_prompt = generate_homework_prompt(
                st.session_state.current_topic,
                st.session_state.client,
                selected_language
            )

if st.session_state.homework_prompt:
    st.info(f"**Assigned Task:**\n\n{st.session_state.homework_prompt}")

with st.form("assignment_eval_form"):
    student_submission = st.text_area("Your Text Answer / Notes (Optional if uploading file)", placeholder="Type your answer here or leave blank if uploading a document/image...")
    uploaded_file = st.file_uploader("Upload Homework File (PDF, JPG, PNG)", type=["pdf", "png", "jpg", "jpeg"])
    
    submit_evaluation = st.form_submit_button("Submit Answer for Evaluation 📋")

    if submit_evaluation:
        if "client" not in st.session_state:
            st.error("Client not initialized. Check your API key configuration.")
        elif not st.session_state.homework_prompt:
            st.warning("Please click 'Get Homework Task from Tutor' first to receive your assignment.")
        elif not student_submission and not uploaded_file:
            st.warning("Please provide a text answer or upload a homework file.")
        else:
            with st.spinner("Evaluating student response and file contents..."):
                try:
                    contents_payload = []
                    eval_query = (
                        f"Act as an expert professor evaluating a student homework submission for the topic '{st.session_state.current_topic}' in {selected_language}.\n\n"
                        f"ASSIGNED TASK:\n{st.session_state.homework_prompt}\n\n"
                    )
                    if student_submission:
                        eval_query += f"STUDENT'S TEXT SUBMISSION:\n{student_submission}\n\n"
                    
                    contents_payload.append(eval_query)

                    if uploaded_file is not None:
                        file_bytes = uploaded_file.read()
                        file_mime = uploaded_file.type
                        
                        if "image" in file_mime:
                            image_obj = Image.open(BytesIO(file_bytes))
                            contents_payload.append(image_obj)
                        elif "pdf" in file_mime:
                            pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
                            contents_payload.append(pdf_part)

                    contents_payload.append(
                        "Provide a comprehensive evaluation structured as follows:\n"
                        "1. **Score / Grade**: (e.g., Score: 85/100 or Letter Grade A/B/C)\n"
                        "2. **Executive Summary**: A brief overview of how well the student's submission addressed the task.\n"
                        "3. **Strengths**: What the student answered correctly or explained well.\n"
                        "4. **Areas for Improvement**: Specific gaps, omissions, or conceptual inaccuracies.\n"
                        "5. **Constructive Feedback & Correct Guidance**: Actionable advice on how the student can improve."
                    )

                    eval_response = safe_generate_content(st.session_state.client, 'gemini-3.6-flash', contents_payload)
                    
                    st.markdown("---")
                    st.markdown("### 🏆 Evaluation Report")
                    st.markdown(eval_response.text)
                except Exception as e:
                    st.error(f"Error evaluating assignment: {e}")

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
    f"👀 Total Visitors: <b>{visitor_count}</b>"
    f"</div>",
    unsafe_allow_html=True
)
