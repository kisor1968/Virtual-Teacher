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

# Initialize Session State Variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_topic" not in st.session_state:
    st.session_state["current_topic"] = "General Studies"

# ==========================================
# Sidebar Configuration & RAG Materials
# ==========================================
st.sidebar.title("📚 Classroom Controls")

# API Key Setup
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
if api_key:
    try:
        genai.configure(api_key=api_key)
        st.session_state["client"] = genai
        st.sidebar.success("API Key configured successfully!")
    except Exception as e:
        st.sidebar.error(f"Error configuring API key: {e}")

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

# Helper function for safe generation
def safe_generate_content(client, model_name, contents):
    model = client.GenerativeModel(model_name)
    return model.generate_content(contents)

# ==========================================
# Main Application Layout
# ==========================================
st.title("🧑‍🏫 Virtual AI Classroom & Tutor")
st.markdown("Welcome to your interactive session! Ask questions, review notes, draw on the whiteboard, and test your knowledge.")

# Topic Setter
current_topic = st.text_input("Today's Lesson Topic / Subject:", value=st.session_state["current_topic"])
st.session_state["current_topic"] = current_topic

# Display Chat History
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
user_prompt = st.chat_input("Ask your AI professor a question...")

if user_prompt:
    if "client" not in st.session_state:
        st.warning("Please enter your Gemini API Key in the sidebar first.")
    else:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.spinner("Professor is preparing the lesson explanation..."):
            try:
                # Build content payload with RAG files if available
                payload = []
                
                # Process uploaded RAG documents
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.read()
                        if uploaded_file.type == "application/pdf":
                            payload.append({
                                "mime_type": "application/pdf",
                                "data": file_bytes
                            })
                        else:
                            payload.append(uploaded_file.getvalue().decode("utf-8"))
                
                # System and contextual instructions
                system_instruction = (
                    f"You are an expert AI Professor teaching the topic '{current_topic}' "
                    f"in {selected_language}. Incorporate insights and details from any provided study documents (RAG materials) "
                    f"when answering student questions accurately."
                )
                
                payload.append(system_instruction)
                
                # Include chat history context
                chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"][-5:]])
                payload.append(f"Recent Discussion Context:\n{chat_history_str}")
                payload.append(f"Student Question: {user_prompt}")
                
                # Generate response from Gemini
                response = safe_generate_content(st.session_state["client"], 'gemini-2.5-flash', payload)
                answer_text = response.text
                
                st.session_state["messages"].append({"role": "assistant", "content": answer_text})
                with st.chat_message("assistant"):
                    st.markdown(answer_text)
                    
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

# ==========================================
# Interactive Whiteboard & Sketchpad Section
# ==========================================
st.markdown("---")
st.subheader("🎨 Interactive Whiteboard & Sketchpad")
st.markdown("Draw diagrams, geometric shapes, or handwritten equations for the AI professor to review.")

col1, col2 = st.columns([1, 1])
with col1:
    stroke_width = st.slider("Stroke Width", 1, 25, 3)
with col2:
    stroke_color = st.color_picker("Stroke Color", "#000000")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#ffffff",
    update_streamlit=True,
    height=350,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas",
)

if st.button("Ask Professor to Analyze Drawing"):
    if "client" not in st.session_state:
        st.error("Client not initialized. Check your API key configuration in the sidebar.")
    elif canvas_result.image_data is not None:
        with st.spinner("Professor is analyzing your sketch..."):
            try:
                img_data = canvas_result.image_data
                # Convert NumPy array directly to PIL Image cleanly without OpenCV dependency
                image_obj = Image.fromarray(img_data.astype('uint8'))
                
                drawing_prompt = [
                    image_obj,
                    f"Analyze this student's whiteboard drawing or handwritten formula regarding the topic '{st.session_state['current_topic']}'. "
                    f"Provide constructive academic feedback, correct any errors, and explain the concept clearly in {selected_language}."
                ]
                
                response = safe_generate_content(st.session_state["client"], 'gemini-2.5-flash', drawing_prompt)
                
                st.markdown("### 🧑‍🏫 Professor's Feedback on Your Sketch")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error analyzing whiteboard sketch: {e}")
    else:
        st.warning("Please draw something on the canvas first before requesting analysis.")

# ==========================================
# Dynamic AI Quiz Section
# ==========================================
st.markdown("---")
st.subheader("📝 Dynamic Lesson Quiz")
st.markdown("Test your understanding with questions generated dynamically by your AI professor based on today's session.")

if st.button("Generate Quiz for This Topic") or "dynamic_quiz" not in st.session_state:
    if "client" in st.session_state:
        with st.spinner("Professor is creating a customized quiz for you..."):
            chat_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.get("messages", [])[-6:]])
            
            quiz_prompt = f"""
            Based on the current topic '{st.session_state['current_topic']}' and the following recent discussion context:
            {chat_context}
            
            Generate 3 multiple-choice questions (MCQs) to test the student's mastery of the material.
            Format your response cleanly with clear questions, options, and explanations.
            """
            try:
                response = safe_generate_content(st.session_state["client"], 'gemini-2.5-flash', quiz_prompt)
                st.session_state["dynamic_quiz"] = response.text
                st.success("Quiz generated successfully!")
            except Exception as e:
                st.error(f"Failed to generate quiz: {e}")
    else:
        st.warning("Please configure your API key in the sidebar first.")

if "dynamic_quiz" in st.session_state:
    st.markdown("### Assessment Questions:")
    st.markdown(st.session_state["dynamic_quiz"])
