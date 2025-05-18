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
# 🔧 Set Page Config (Must be the first Streamlit command)
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
with open("emontions_advice.json", "r") as f:
    advice_data = json.load(f)

# --------------------------------------------
# 📷 Image Processing & Analysis
# --------------------------------------------
def analyze_emotion_and_display(image):
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, img_bgr)
    emotions = DeepFace.analyze(img_path=temp_path, actions=['emotion'])
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
    analysis = DeepFace.analyze(img_path=temp_path, actions=['emotion', 'age', 'gender'])
    df = pd.json_normalize(analysis)
    return df

# --------------------------------------------
# 📊 Visualization Functions
# --------------------------------------------
def create_hierarchical_tree(df):
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
        textinfo="label+value",
        textfont=dict(size=16, family="Arial", color="black"),
        insidetextfont=dict(size=16, color="black"),
        textposition="middle center"
    ))

    fig.update_layout(title="Emotion And Personal Info From Photo")
    return fig

def plot_emotion_confidence(df):
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
        title="Emotion Confidence Levels with Emojis",
        xaxis_title="Confidence Level",
        yaxis_title="Emotion",
        barmode='stack',
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
        return "Sorry, I don’t have advice for that feeling yet."
    entry = random.choice(entries)
    advice = random.choice(entry["advice"])
    prompt = (
        f"User feels {emotion} and said: '{user_response}'.\n"
        f"Advice: \"{advice}\"\n"
        f"Rewrite this as a single empathetic sentence directly addressing the user's situation. "
        f"Respond only with the rephrased line in quotes, no explanation or questions."
    )
    try:
        response = llm(prompt, max_new_tokens=60, temperature=0.7)
        return response.strip().strip('"')
    except Exception as e:
        return f"Error generating advice: {e}"

def chat(emotion, user_input, history):
    if user_input.lower() in ["exit", "goodbye", "stop"]:
        final_msg = (
            "Captain Feels🤖: Thank you for exploring. If you liked it, share & like my LinkedIn post 🌟\n\n"
            "🤖 Keep shining. Take care 🌸"
        )
        return history + [[f"You: {user_input}", final_msg]], ""
    advice = get_advice_from_json(emotion, user_input)
    new_pair = [f"You: {user_input}", f"💡 Advice: {advice}"]
    return history + [new_pair], ""

def respond(emotion, user_text, chat_hist, questions, q_idx):
    if user_text.lower() in ["exit", "goodbye", "stop"]:
        updated_chat, _ = chat(emotion, user_text, chat_hist)
        return updated_chat, "", questions, q_idx

    updated_chat, _ = chat(emotion, user_text, chat_hist)

    if q_idx < len(questions):
        next_q = questions[q_idx]
        updated_chat.append([f"Captain Feels 🤖: Q{q_idx + 1}: {next_q}", ""])
        q_idx += 1
    else:
        updated_chat.append(["Captain Feels 🤖: You’ve finally completed the journey. You’re doing your best 🌈.\n\n"
                             "If you enjoyed this experience, please like and share my LinkedIn post 🌟. 🤖 Keep shining, take care 🌸, and thank you for sharing! 🌈", ""])

    return updated_chat, "", questions, q_idx

# --------------------------------------------
# 🔊 Audio Generation
# --------------------------------------------
@st.cache_data
def generate_welcome_audio():
    try:
        welcome_text = (
            "Hello! I’m Captain Feels — your virtual Emo Bot and Emotion Assistant. Please be patient as we go through the process. First, you’ll hear a short audio message. Then, upload or drag and drop your image for analysis. Finally, I’ll guide you with personalized advice and a few interactive questions."
        )
        tts = gTTS(text=welcome_text, lang='en', tld='co.uk')  # British English for neutral/male-leaning voice
        _, audio_path = tempfile.mkstemp(suffix=".mp3")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.warning(f"Failed to generate welcome audio: {e}. Continuing without audio.")
        return None

# --------------------------------------------
# 🎨 Streamlit App UI
# --------------------------------------------
def main():
    st.title("😎 EmoGuide: Emotion-Aware Conversations with Captain Feels")
    st.markdown("## 🚩 Play the Audio and Start the Journey with Patience")

    # Play welcome audio
    welcome_audio_path = generate_welcome_audio()
    if welcome_audio_path:
        st.audio(welcome_audio_path, format="audio/mp3", start_time=0)
    else:
        st.info("Audio generation skipped due to TTS initialization failure.")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.emotion = "neutral"
        st.session_state.questions = []
        st.session_state.question_index = 0
        st.session_state.analysis_done = False

    # Image upload
    st.markdown("### 📸 Upload or Capture Image")
    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if st.button("Analyze Emotion 🧠"):
            with st.spinner("Analyzing image..."):
                # Process image
                result_img = analyze_emotion_and_display(image_rgb)
                df = analyze_full_info(image_rgb)
                st.session_state.emotion = df['dominant_emotion'][0]
                st.session_state.questions = get_default_questions(st.session_state.emotion)
                st.session_state.question_index = 1
                st.session_state.chat_history = [[f"", f"Captain Feels 🤖: Q1: {st.session_state.questions[0]}"]]
                st.session_state.analysis_done = True

                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result_img, caption="Emotion Detection Output", use_column_width=True)
                with col2:
                    st.plotly_chart(create_hierarchical_tree(df), use_container_width=True)
                st.plotly_chart(plot_emotion_confidence(df), use_container_width=True)
                st.markdown(f"### ✅ Captain Feels: I sense you're feeling **{st.session_state.emotion.upper()}**.")

    # Chatbot interaction
    if st.session_state.analysis_done:
        st.markdown("##### 💬 Your Message")
        user_input = st.text_input("Type your response or 'exit' to end...", key="user_input")
        
        if st.button("Send"):
            if user_input:
                with st.spinner("Generating response..."):
                    st.session_state.chat_history, _, st.session_state.questions, st.session_state.question_index = respond(
                        st.session_state.emotion,
                        user_input,
                        st.session_state.chat_history,
                        st.session_state.questions,
                        st.session_state.question_index
                    )

        # Display chat history
        for user_msg, bot_msg in st.session_state.chat_history:
            if user_msg:
                st.markdown(f"**You:** {user_msg}")
            if bot_msg:
                st.markdown(f"**Captain Feels 🤖:** {bot_msg}")

        st.markdown("##### 🚩 To achieve better results, please write well-structured prompts with explanations.")

if __name__ == "__main__":
    main()
