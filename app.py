import streamlit as st
import os
import shutil
from core.processor import ImmersionProcessor

st.set_page_config(page_title="Hayaku Local", page_icon="🇯🇵", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .transcript-container { background-color: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; }
    .timestamp { color: #8b949e; font-family: monospace; font-size: 14px; margin-right: 10px; }
    .word-link { color: #58a6ff; text-decoration: none; padding: 2px 4px; border-radius: 4px; transition: all 0.2s; cursor: pointer; font-size: 20px; display: inline-block; }
    .word-link:hover { background-color: #1f6feb; color: white; }
    .romaji { color: #8b949e; font-size: 12px; display: block; margin-bottom: 2px; }
    .segment-block { margin-bottom: 25px; border-bottom: 1px solid #21262d; padding-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

if 'processor' not in st.session_state:
    st.session_state.processor = ImmersionProcessor()
if 'lesson' not in st.session_state:
    st.session_state.lesson = None

st.sidebar.title("🇯🇵 Hayaku Local")
uploaded_file = st.sidebar.file_uploader("Upload Media", type=["mp4", "mp3", "wav", "m4a"])
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
if uploaded_file:
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.title(f"📺 {uploaded_file.name}")
    if uploaded_file.name.endswith(".mp4"):
        st.video(file_path)
    else:
        st.audio(file_path)

    if st.button("🚀 Start Immersion Processing", key="btn_process", use_container_width=True):
        with st.spinner("Transcribing and Tokenizing..."):
            st.session_state.lesson = st.session_state.processor.process_media(file_path, uploaded_file.name)
            st.success("Processing Complete!")

    if st.session_state.lesson:
        st.subheader("📝 Interactive Transcript")
        html_output = '<div class="transcript-container">'
        for seg in st.session_state.lesson.segments:
            html_output += f'<div class="segment-block"><span class="timestamp">[{seg.start:.2f}s]</span>'
            for token in seg.tokens:
                html_output += f'<a href="{token.url}" target="_blank" class="word-link"><span class="romaji">{token.reading}</span>{token.text}</a> '
            html_output += '</div>'
        html_output += '</div>'
        st.markdown(html_output, unsafe_allow_html=True)

if st.sidebar.button("🗑️ Clear Session", key="btn_clear"):
    st.session_state.lesson = None
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    st.rerun()

