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

# --------------------------------------------
# 📷 Image Processing & Analysis
# --------------------------------------------
def analyze_emotion_and_display(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, img_bgr)
    try:
        emotions = DeepFace.analyze(img_path=temp_path, actions=['emotion'])
    except Exception as e:
        st.error(f"Emotion analysis failed: {e}")
        return image
    padding = 50
    img_rgb_padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    for face in emotions:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        dominant_emotion = face['dominant_emotion']
        emotion_percentage = face['emotion'][dominant_emotion]

        text = f"{dominant_emotion}: {emotion_percentage:.2f}%"
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_width, text_height = text_size
        pad = 20
        x += padding
        y += padding
        rect_width = max(w, text_width) + pad * 2
        rect_x1 = x - pad
        rect_x2 = rect_x1 + rect_width
        rect_y1 = y - text_height - pad - 10
        rect_y2 = rect_y1 + text_height + pad

        cv2.rectangle(img_rgb_padded, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        cv2.rectangle(img_rgb_padded, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 4)
        text_x = rect_x1 + (rect_width - text_width) // 2
        text_y = rect_y1 + (text_height + pad) // 2 + baseline
        cv2.putText(img_rgb_padded, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        face_rect_x1 = x - pad
        face_rect_x2 = max(x + w + pad, rect_x2)
        cv2.rectangle(img_rgb_padded, (face_rect_x1, y), (face_rect_x2, y + h), (255, 0, 0), 4)

    return img_rgb_padded

def analyze_full_info(image):
    temp_path = "temp_image.jpg"
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, image_bgr)
    try:
        analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion', 'age', 'gender'])
        df = pd.json_normalize(analysis)
        return df
    except Exception as e:
        st.error(f"Full analysis failed: {e}")
        return pd.DataFrame()

# --------------------------------------------
# 📊 Visualization Functions
# --------------------------------------------
def create_hierarchical_tree(df):
    if df.empty:
        return None
    labels = [
        "Person Info",
        f"CURRENT EMOTION: {df['dominant_emotion'][0].upper()}",
        f"FACE CONFIDENCE: {df['face_confidence'][0] * 100:.2f}%",
        f"APPROXIMATE AGE: {df['age'][0]}",
        f"GENDER: {df['dominant_gender'][0].upper()}"
    ]
    parents = ["", "Person Info", "Person Info", "Person Info", "Person Info"]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        marker=dict(colors=[0, 1, 2, 3, 4], colorscale='Sunset'),
        textinfo="label",
        textfont=dict(size=16, family="Arial", color="black"),
        insidetextfont=dict(size=16, color="black"),
        textposition="middle center"
    ))
    fig.update_layout(title="Emotion and Personal Info")
    return fig

def plot_emotion_confidence(df):
    if df.empty:
        return None
    df_filter = df[['emotion.angry', 'emotion.disgust', 'emotion.fear', 'emotion.happy', 'emotion.sad', 'emotion.surprise', 'emotion.neutral']]
    df_filter.columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    emotion_to_emoji = {
        'angry': '😠', 'disgust': '😷', 'fear': '😨',
        'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
    }
    emotion_to_color = {
        'angry': '#B22222', 'disgust': '#8B4513', 'fear': '#800080',
        'happy': '#006400', 'sad': '#1E90FF', 'surprise': '#FF8C00', 'neutral': '#696969'
    }

    sorted_columns = df_filter.iloc[0].sort_values(ascending=False)
    fig = go.Figure()
    for emo in sorted_columns.index:
        fig.add_trace(go.Bar(
            x=[sorted_columns[emo]],
            y=[f"{emotion_to_emoji[emo]} {emo}"],
            orientation='h',
            marker=dict(color=emotion_to_color[emo]),
            name=emo,
            text=[f"{sorted_columns[emo]:.2f}%"],
            textposition='outside'
        ))

    fig.update_layout(
        title="Emotion Confidence Levels",
        xaxis_title="Confidence Level (%)",
        yaxis_title="Emotion",
        height=400
    )
    fig.update_xaxes(range=[0, 110])
    return fig

# --------------------------------------------
# 💬 NLP and Chat Logic
# --------------------------------------------
def get_default_questions(emotion):
    questions = [entry["question"] for entry in advice_data if entry["emotion"] == emotion]
    return questions[:5] if questions else ["Can you share how you're feeling today?"]

def get_advice_from_json(emotion, user_response):
    entries = [entry for entry in advice_data if entry["emotion"] == emotion]
    if not entries:
        return "I don’t have specific advice for this emotion yet, but I’m here to listen."
    entry = random.choice(entries)
    advice = random.choice(entry["advice"])
    prompt = (
        f"User feels {emotion} and said: '{user_response}'.\n"
        f"Advice: \"{advice}\"\n"
        f"Rewrite as a single empathetic sentence directly addressing the user's situation. "
        f"Respond only with the rephrased line in quotes, no explanation or questions."
    )
    try:
        response = llm(prompt, max_new_tokens=60, temperature=0.7)
        return response.strip().strip('"')
    except Exception as e:
        return f"Error generating advice: {e}"

# --------------------------------------------
# 🔊 Audio Generation
# --------------------------------------------
@st.cache_data
def generate_welcome_audio():
    try:
        welcome_text = (
            "Hello! I’m Captain Feels, your virtual Emotion Assistant. "
            "First, listen to this message. Then, upload an image for emotion analysis. "
            "After that, I’ll ask you questions one by one, offer advice, and guide you through the journey."
        )
        tts = gTTS(text=welcome_text, lang='en', tld='co.uk')
        _, audio_path = tempfile.mkstemp(suffix=".mp3")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.warning(f"Failed to generate audio: {e}. Continuing without audio.")
        return None

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

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'audio'
        st.session_state.chat_history = []
        st.session_state.emotion = "neutral"
        st.session_state.questions = []
        st.session_state.question_index = 0
        st.session_state.analysis_done = False
        st.session_state.user_input = ""

    # Image upload (only after audio)
    if st.session_state.stage == 'audio':
        if st.button("Proceed to Image Upload"):
            st.session_state.stage = 'image'

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

    # Chatbot interaction
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
                st.markdown(f"**Captain Feels 🤖: Q{st.session_state.question_index + 1}:** {st.session_state.questions[st.session_state.question_index]}")
                user_input = st.text_input("Your response (or type 'exit' to end)...", key=f"user_input_{st.session_state.question_index}")
                
                if st.button("Send Response"):
                    if user_input:
                        with st.spinner("Generating advice..."):
                            st.session_state.chat_history.append([f"You: {user_input}", ""])
                            if user_input.lower() in ["exit", "goodbye", "stop"]:
                                st.session_state.chat_history.append([
                                    "",
                                    ("Captain Feels 🤖: Thank you for exploring! If you liked it, share & like my LinkedIn post 🌟\n\n"
                                     "🤖 Keep shining. Take care 🌸")
                                ])
                                st.session_state.stage = 'done'
                            else:
                                advice = get_advice_from_json(st.session_state.emotion, user_input)
                                st.session_state.chat_history[-1][1] = f"💡 Captain Feels 🤖: {advice}"
                                st.session_state.question_index += 1
                                if st.session_state.question_index < len(st.session_state.questions):
                                    st.session_state.chat_history.append([
                                        f"Captain Feels 🤖: Q{st.session_state.question_index + 1}: {st.session_state.questions[st.session_state.question_index]}",
                                        ""
                                    ])
                                else:
                                    st.session_state.chat_history.append([
                                        "",
                                        ("Captain Feels 🤖: You’ve completed the journey! You’re doing your best 🌈.\n\n"
                                         "If you enjoyed this, please like and share my LinkedIn post 🌟. "
                                         "🤖 Keep shining, take care 🌸, and thank you for sharing! 🌈")
                                    ])
                                    st.session_state.stage = 'done'
                            st.experimental_rerun()
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
                        st.experimental_rerun()

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
