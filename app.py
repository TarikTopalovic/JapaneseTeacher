import streamlit as st
import os
import html
import pandas as pd
import yt_dlp
from core.processor import ImmersionProcessor
from core.database import ImmersionDB

st.set_page_config(page_title='速く — Hayaku Pro', layout='wide', page_icon='🎌')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+JP:wght@300;400;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', 'Noto Sans JP', sans-serif; }
.stApp { background: #080c12; color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0e16 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px;
    border-radius: 8px;
    transition: all 0.2s;
    display: block;
    font-weight: 500;
    color: #8b99b0;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.05); color: #e2e8f0; }

/* ── Logo badge ── */
.logo-badge {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 0 20px 0;
}
.logo-circle {
    width: 42px; height: 42px; border-radius: 50%;
    background: linear-gradient(135deg, #e63946, #c1121f);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; box-shadow: 0 0 20px rgba(230,57,70,0.4);
    flex-shrink: 0;
}
.logo-text { line-height: 1.1; }
.logo-title { font-size: 16px; font-weight: 700; color: #f0f4f8; letter-spacing: 0.5px; }
.logo-sub { font-size: 11px; color: #4a5568; text-transform: uppercase; letter-spacing: 1.5px; }

/* ── Section headers ── */
.section-header {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.5px; color: #4a5568; margin: 24px 0 10px 0;
}

/* ── Panel cards ── */
.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #6b7a99 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(230,57,70,0.15) !important;
    color: #e63946 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e63946, #c1121f) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 12px rgba(230,57,70,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(230,57,70,0.45) !important;
}

/* ── Text input ── */
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 13px !important;
}
.stTextInput input:focus { border-color: #e63946 !important; }

/* ── Transcript scroll ── */
.transcript-wrap {
    height: 72vh;
    overflow-y: auto;
    padding-right: 6px;
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.1) transparent;
}
.transcript-wrap::-webkit-scrollbar { width: 4px; }
.transcript-wrap::-webkit-scrollbar-track { background: transparent; }
.transcript-wrap::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }

/* ── Segment ── */
.seg-sentence {
    font-size: 11px;
    color: #4a5568;
    font-family: 'Noto Sans JP', sans-serif;
    margin-bottom: 6px;
    padding-left: 4px;
    border-left: 2px solid rgba(230,57,70,0.3);
    padding-left: 8px;
}
.seg-translation {
    font-size: 12px;
    color: #8b99b0;
    margin-bottom: 8px;
}

/* ── Word cards ── */
.words-row { display: flex; flex-wrap: wrap; gap: 4px; align-items: flex-end; margin-bottom: 18px; }

.word-card {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    padding: 6px 8px 5px;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    cursor: pointer;
    position: relative;
    transition: all 0.18s ease;
    text-decoration: none;
}
.word-card:hover {
    background: rgba(230,57,70,0.08);
    border-color: rgba(230,57,70,0.4);
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(230,57,70,0.15);
}
.word-reading {
    font-size: 9px;
    color: #4a5568;
    font-family: 'Inter', sans-serif;
    text-transform: lowercase;
    letter-spacing: 0.3px;
    margin-bottom: 2px;
    white-space: nowrap;
}
.word-surface {
    font-size: 22px;
    font-weight: 700;
    color: #f0f4f8;
    font-family: 'Noto Sans JP', sans-serif;
    line-height: 1;
}

/* ── Tooltip ── */
.word-card .tip {
    visibility: hidden; opacity: 0;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%; transform: translateX(-50%);
    width: 200px;
    background: #131920;
    border: 1px solid rgba(230,57,70,0.35);
    border-radius: 10px;
    padding: 12px;
    z-index: 999;
    transition: opacity 0.15s ease;
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
    pointer-events: none;
}
.word-card .tip::after {
    content: '';
    position: absolute;
    top: 100%; left: 50%; transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: rgba(230,57,70,0.35);
}
.word-card:hover .tip { visibility: visible; opacity: 1; }
.tip-word { font-size: 18px; font-weight: 700; color: #e63946; font-family: 'Noto Sans JP'; }
.tip-reading { font-size: 11px; color: #4a5568; margin: 2px 0 6px; }
.tip-divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 6px 0; }
.tip-meaning { font-size: 13px; color: #f6ad55; font-weight: 500; }
.tip-link { font-size: 10px; color: #4a5568; margin-top: 6px; display: block; text-decoration: none; }

/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 300px; text-align: center; color: #2d3748;
}
.empty-state .icon { font-size: 52px; margin-bottom: 16px; opacity: 0.4; }
.empty-state p { font-size: 14px; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #4a5568 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #f0f4f8 !important; font-size: 28px !important; font-weight: 700 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.07); }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #e63946 !important; }
</style>
""", unsafe_allow_html=True)

# ── State ──────────────────────────────────────────────────────────────────
if 'processor' not in st.session_state: st.session_state.processor = ImmersionProcessor()
if 'db'        not in st.session_state: st.session_state.db        = ImmersionDB()
if 'lesson'    not in st.session_state: st.session_state.lesson    = None
if 'jump_time' not in st.session_state: st.session_state.jump_time = 0

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-badge">
        <div class="logo-circle">🎌</div>
        <div class="logo-text">
            <div class="logo-title">HAYAKU PRO</div>
            <div class="logo-sub">速く · Immersion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    page = st.radio('', ['🎬  Immersion Room', '📊  Vocab Analytics', '⚙️  Settings'], label_visibility='collapsed')

    st.markdown('---')
    lesson_count = len(st.session_state.lesson) if st.session_state.lesson else 0
    if lesson_count:
        st.markdown(f'<div style="font-size:12px;color:#4a5568;">📝 {lesson_count} segments loaded</div>', unsafe_allow_html=True)

    vocab_count = len(st.session_state.db.get_all(limit=1_000_000))
    st.markdown(f'<div style="font-size:12px;color:#4a5568;margin-top:6px;">🗂 {vocab_count} words in bank</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#2d3748;margin-top:16px;">v3.0 · Jamdict · Whisper medium</div>', unsafe_allow_html=True)

# ── Page: Immersion Room ───────────────────────────────────────────────────
if '🎬' in page:
    col_vid, col_txt = st.columns([1.15, 1], gap='large')

    with col_vid:
        st.markdown('<div class="section-header">Media Source</div>', unsafe_allow_html=True)
        tab_yt, tab_local = st.tabs(['🌐  YouTube Link', '📁  Local File'])

        with tab_yt:
            yt_url = st.text_input('', placeholder='Paste YouTube URL — anime, VTuber, podcast…', label_visibility='collapsed')
            if yt_url:
                st.video(yt_url, start_time=st.session_state.jump_time)
                if st.button('🚀  Analyze Audio', key='btn_yt'):
                    with st.spinner('Downloading audio stream…'):
                        ydl_opts = {'format': 'bestaudio/best', 'outtmpl': 'temp_yt_audio.%(ext)s', 'quiet': True, 'no_warnings': True}
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info       = ydl.extract_info(yt_url, download=True)
                            audio_path = ydl.prepare_filename(info)
                            video_title= info.get('title', 'YouTube_Video')
                    with st.spinner('Transcribing · tracking speakers · translating sentences · Jamdict lookup…'):
                        st.session_state.lesson = st.session_state.processor.process_media(audio_path, video_title)
                        if os.path.exists(audio_path): os.remove(audio_path)
                        st.rerun()

        with tab_local:
            up = st.file_uploader('', type=['mp4', 'mp3', 'wav'], label_visibility='collapsed')
            if up:
                temp_path = f'temp_{up.name}'
                with open(temp_path, 'wb') as f: f.write(up.getbuffer())
                ext = up.name.rsplit('.', 1)[-1].lower()
                if ext in ('mp3', 'wav'):
                    st.audio(temp_path)
                else:
                    st.video(temp_path, start_time=st.session_state.jump_time)
                if st.button('🚀  Analyze Media', key='btn_local'):
                    with st.spinner('Transcribing · tracking speakers · translating sentences · dictionary lookup…'):
                        st.session_state.lesson = st.session_state.processor.process_media(temp_path, up.name)
                        st.rerun()

    with col_txt:
        st.markdown('<div class="section-header">Interactive Transcript</div>', unsafe_allow_html=True)

        if st.session_state.lesson:
            st.markdown('<div class="transcript-wrap">', unsafe_allow_html=True)
            for idx, seg in enumerate(st.session_state.lesson):
                ts  = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
                ts_col, words_col = st.columns([0.14, 0.86])

                with ts_col:
                    if st.button(ts, key=f'jump_{idx}'):
                        st.session_state.jump_time = int(seg['start'])
                        st.rerun()

                with words_col:
                    raw_sentence = html.escape(seg.get('text', ''))
                    speaker = html.escape(seg.get('speaker', 'SPEAKER_1'))
                    translation = html.escape(seg.get('translation', '—'))
                    seg_html = (
                        f'<div class="seg-sentence">[{speaker}] {raw_sentence}</div>'
                        f'<div class="seg-translation">{translation}</div>'
                        f'<div class="words-row">'
                    )
                    for t in seg['tokens']:
                        s_text    = html.escape(t['text'])
                        s_reading = html.escape(t['reading'])
                        s_meaning = html.escape(t.get('meaning', '—'))
                        s_url     = html.escape(t.get('url', '#'))
                        seg_html += (
                            f'<div class="word-card">'
                            f'  <span class="word-reading">{s_reading}</span>'
                            f'  <span class="word-surface">{s_text}</span>'
                            f'  <div class="tip">'
                            f'    <div class="tip-word">{s_text}</div>'
                            f'    <div class="tip-reading">{s_reading}</div>'
                            f'    <hr class="tip-divider">'
                            f'    <div class="tip-meaning">{s_meaning}</div>'
                            f'    <a class="tip-link" href="{s_url}" target="_blank">🔍 Open in Jisho</a>'
                            f'  </div>'
                            f'</div>'
                        )
                    seg_html += '</div>'
                    st.markdown(seg_html, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="icon">🎌</div>
                <p>Paste a YouTube link or upload a file<br>to generate your interactive transcript.</p>
            </div>
            """, unsafe_allow_html=True)

# ── Page: Vocab Analytics ──────────────────────────────────────────────────
elif '📊' in page:
    st.markdown('<div class="section-header" style="margin-top:8px;">Vocabulary Bank</div>', unsafe_allow_html=True)
    data = st.session_state.db.get_all(limit=1_000_000)

    if data:
        df = pd.DataFrame(data)
        df = df[['word', 'reading', 'meaning', 'count']].rename(columns={
            'word': 'Word', 'reading': 'Reading',
            'meaning': 'Meaning', 'count': 'Seen',
        })

        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Total Words',       len(df))
        c2.metric('High Frequency',    len(df[df['Seen'] > 3]),  help='Seen more than 3 times')
        c3.metric('Seen Once',         len(df[df['Seen'] == 1]))
        c4.metric('Top Word Count',    int(df['Seen'].max()))

        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

        col_search, col_sort = st.columns([3, 1])
        with col_search:
            search = st.text_input('', placeholder='🔍  Filter words…', label_visibility='collapsed')
        with col_sort:
            sort_by = st.selectbox('', ['Most Seen', 'Alphabetical'], label_visibility='collapsed')

        if search:
            mask = (df['Word'].str.contains(search, na=False) |
                    df['Reading'].str.contains(search, na=False) |
                    df['Meaning'].str.contains(search, case=False, na=False))
            df = df[mask]

        df = df.sort_values('Seen', ascending=False) if sort_by == 'Most Seen' else df.sort_values('Word')
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📖</div>
            <p>No words logged yet.<br>Analyze some media to start building your vocab bank.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Page: Settings ─────────────────────────────────────────────────────────
elif '⚙️' in page:
    st.markdown('<div class="section-header" style="margin-top:8px;">System Settings</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**🤖 Whisper**')
        st.code(
            f"Model:   {os.getenv('WHISPER_MODEL', 'medium')}\n"
            f"Device:  {os.getenv('WHISPER_DEVICE', 'cuda')}\n"
            f"Compute: {os.getenv('WHISPER_COMPUTE', 'int8_float16')}",
            language='yaml'
        )
        st.caption('Set via environment variables before launching.')

    with c2:
        st.markdown('**🗄️ Database**')
        db_path = st.session_state.db.path
        st.code(f"Path:   {db_path}\nExists: {db_path.exists()}", language='yaml')

    st.markdown('---')
    st.markdown('**🗑️ Reset Vocab Bank**')
    st.caption('This permanently deletes all logged vocabulary. Cannot be undone.')
    if st.button('⚠️  Clear All Vocabulary', type='primary'):
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.execute('DELETE FROM vocabulary')
        st.success('Vocabulary bank cleared.')
        st.rerun()
