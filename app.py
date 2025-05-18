import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from deepface import DeepFace
import json
import random
from ctransformers import AutoModelForCausalLM
from gtts import gTTS
import tempfile
import os

# --------------------------------------------
# 🔧 Initialize Session State
# --------------------------------------------
def initialize_session_state():
    if 'stage' not in st.session_state:
        st.session_state.stage = 'audio'
        st.session_state.chat_history = []
        st.session_state.emotion = "neutral"
        st.session_state.questions = []
        st.session_state.question_index = 0
        st.session_state.analysis_done = False
        st.session_state.user_input = ""

# Call initialization immediately
initialize_session_state()

# --------------------------------------------
# 🔧 Set Page Config
# --------------------------------------------
st.set_page_config(page_title="EmoGuide: Emotion-Aware Conversations", page_icon="😎")

# --------------------------------------------
# 🔧 Initialize Models
# --------------------------------------------
@st.cache_resource
def load_llm():
    try:
        return AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7B-Chat-GGUF",
            model_file="llama-2-7b-chat.Q4_K_M.gguf",
            model_type="llama",
            gpu_layers=0
        )
    except Exception as e:
        st.error(f"Failed to load LLaMA model: {e}")
        return None

llm = load_llm()

# --------------------------------------------
# 📂 Load JSON Advice Data
# --------------------------------------------
try:
    with open("emontions_advice.json", "r") as f:
        advice_data = json.load(f)
except FileNotFoundError:
    st.error("emontions_advice.json not found. Please ensure the file exists.")
    advice_data = []

# ... (rest of the functions: analyze_emotion_and_display, analyze_full_info, create_hierarchical_tree, plot_emotion_confidence, get_default_questions, get_advice_from_json, generate_welcome_audio remain unchanged)

# --------------------------------------------
# 🎨 Streamlit App UI
# --------------------------------------------
def main():
    st.title("😎 EmoGuide: Emotion-Aware Conversations with Captain Feels")
    st.markdown("### 🚩 Step 1: Listen to the Welcome Audio")

    # Play welcome audio
    welcome_audio_path = generate_welcome_audio()
    if welcome_audio_path:
        st.audio(welcome_audio_path, format="audio/mp3", start_time=0)
    else:
        st.info("Audio generation skipped due to TTS failure.")

    # Image upload (only after audio)
    if st.session_state.stage == 'audio':
        if st.button("Proceed to Image Upload"):
            st.session_state.stage = 'image'
            st.rerun()

    if st.session_state.stage == 'image':
        st.markdown("### 📸 Step 2: Upload Image for Emotion Analysis")
        image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if image_file is not None:
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if st.button("Analyze Emotion 🧠"):
                with st.spinner("Analyzing image..."):
                    result_img = analyze_emotion_and_display(image_rgb)
                    df = analyze_full_info(image_rgb)
                    if not df.empty:
                        st.session_state.emotion = df['dominant_emotion'][0]
                        st.session_state.questions = get_default_questions(st.session_state.emotion)
                        st.session_state.stage = 'chat'
                        st.session_state.analysis_done = True
                        st.session_state.chat_history = [[f"Captain Feels 🤖: Q1: {st.session_state.questions[0]}", ""]]

                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result_img, caption="Emotion Detection Output", use_column_width=True)
                        with col2:
                            fig_tree = create_hierarchical_tree(df)
                            if fig_tree:
                                st.plotly_chart(fig_tree, use_container_width=True)
                        fig_conf = plot_emotion_confidence(df)
                        if fig_conf:
                            st.plotly_chart(fig_conf, use_container_width=True)
                        st.markdown(f"### ✅ Captain Feels: I sense you're feeling **{st.session_state.emotion.upper()}**.")
                    else:
                        st.error("Analysis failed. Please try another image.")

    # Chatbot interaction (using the fixed logic from the previous response)
    if st.session_state.stage == 'chat' and st.session_state.analysis_done:
        st.markdown("### 💬 Step 3: Chat with Captain Feels")
        st.write(f"Detected Emotion: **{st.session_state.emotion.upper()}**")

        if not st.session_state.questions:
            st.warning("No questions found for this emotion. Please try another image.")
        else:
            # Display chat history
            for user_msg, bot_msg in st.session_state.chat_history:
                if user_msg:
                    st.markdown(f"**{user_msg}**")
                if bot_msg:
                    st.markdown(f"**{bot_msg}**")

            # Show current question or final message
            if st.session_state.question_index < len(st.session_state.questions):
                # Ensure the current question is in chat_history
                if not st.session_state.chat_history or st.session_state.chat_history[-1][0] != f"Captain Feels 🤖: Q{st.session_state.question_index + 1}: {st.session_state.questions[st.session_state.question_index]}":
                    st.session_state.chat_history.append([
                        f"Captain Feels 🤖: Q{st.session_state.question_index + 1}: {st.session_state.questions[st.session_state.question_index]}",
                        ""
                    ])
                    st.rerun()

                user_input = st.text_input("Your response (or type 'exit' to end)...", key=f"user_input_{st.session_state.question_index}")
                
                if st.button("Send Response"):
                    if user_input:
                        with st.spinner("Generating advice..."):
                            st.session_state.chat_history[-1][1] = f"You: {user_input}"
                            if user_input.lower() in ["exit", "goodbye", "stop"]:
                                st.session_state.chat_history.append([
                                    "",
                                    ("Captain Feels 🤖: Thank you for exploring! If you liked it, share & like my LinkedIn post 🌟\n\n"
                                     "🤖 Keep shining. Take care 🌸")
                                ])
                                st.session_state.stage = 'done'
                            else:
                                advice = get_advice_from_json(st.session_state.emotion, user_input)
                                st.session_state.chat_history.append([
                                    "",
                                    f"💡 Advice: {advice}"
                                ])
                                st.session_state.question_index += 1
                                if st.session_state.question_index >= len(st.session_state.questions):
                                    st.session_state.chat_history.append([
                                        "",
                                        ("Captain Feels 🤖: You’ve completed the journey! You’re doing your best 🌈.\n\n"
                                         "If you enjoyed this, please like and share my LinkedIn post 🌟. "
                                         "🤖 Keep shining, take care 🌸, and thank you for sharing! 🌈")
                                    ])
                                    st.session_state.stage = 'done'
                            st.rerun()
            else:
                st.markdown("**Captain Feels 🤖:** You've answered all my questions! Type 'exit' to end or share more.")
                user_input = st.text_input("Your response (or type 'exit' to end)...", key="final_input")
                if st.button("Send Final Response"):
                    if user_input.lower() in ["exit", "goodbye", "stop"]:
                        st.session_state.chat_history.append([
                            f"You: {user_input}",
                            ("Captain Feels 🤖: Thank you for exploring! If you liked it, share & like my LinkedIn post 🌟\n\n"
                             "🤖 Keep shining. Take care 🌸")
                        ])
                        st.session_state.stage = 'done'
                        st.rerun()

    if st.session_state.stage == 'done':
        st.markdown("### 🌈 Journey Complete")
        for user_msg, bot_msg in st.session_state.chat_history:
            if user_msg:
                st.markdown(f"**{user_msg}**")
            if bot_msg:
                st.markdown(f"**{bot_msg}**")
        st.stop()

if __name__ == "__main__":
    main()
