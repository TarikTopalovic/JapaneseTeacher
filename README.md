# 🇯🇵 Hayaku Pro

**Local Japanese immersion tool.** Feed it any video or audio — YouTube links, anime clips, podcasts, VTuber streams — and it transcribes speech, tracks speakers, translates full sentences to English, splits text into words, and gives dictionary meanings for each token. No subscriptions. Runs entirely on your machine.

---

## What it does

1. **Transcribes Japanese audio** using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper running locally, GPU-accelerated if available)
2. **Tokenizes each sentence** using Sudachi (a proper Japanese morphological analyzer — handles compound words correctly)
3. **Tracks speaker changes** using local pyannote diarization and labels each segment
4. **Translates full sentences with provider fallback** (Google Translate via `deep-translator` first, then Argos offline fallback)
5. **Looks up token meanings instantly** using Jamdict and Sudachi dictionary forms (base forms), so conjugated words resolve correctly
6. **Shows an interactive transcript** where you can hover any word to see its reading and meaning, and click timestamps to jump in the video
7. **Builds a vocab bank** of every word you've encountered, with frequency counts, so you can track what's appearing most in your immersion content

---

## Project structure

```
hayaku-pro/
├── server.py               # FastAPI backend + API routes
├── static/
│   └── index.html          # Main web UI
├── storage/
│   ├── db/                 # SQLite database location
│   └── media/
│       ├── uploads/        # Temporary uploads while processing
│       └── downloads/      # Temporary YouTube audio downloads
├── start.sh                # One-command launcher
└── core/
    ├── engine.py           # Whisper transcription (faster-whisper)
    ├── speaker_tracking.py # pyannote speaker diarization
    ├── sentence_translation.py # Argos sentence translation
    ├── linguistics.py      # Sudachi tokenizer + Jamdict lookup
    ├── processor.py        # Orchestrates engine → linguistics → database
    ├── database.py         # SQLite vocab bank + lessons persistence
    ├── models.py           # Dataclasses: Token, Segment, Lesson
    ├── config.py           # App settings (DB path, WAL mode)
    └── utils.py            # Shared logger
```

---

## Requirements

### System dependencies

| Tool | Purpose | Install |
|---|---|---|
| **Python 3.10+** | Runtime | [python.org](https://python.org) |
| **CUDA** *(optional)* | GPU acceleration for Whisper | Via your NVIDIA driver |
| **FFmpeg** | Audio extraction from video files | `sudo apt install ffmpeg` / `brew install ffmpeg` |

### Python packages

```
deep-translator
fastapi
faster-whisper
argostranslate
pyannote.audio
jamdict
jamdict-data
sudachipy
sudachidict-core
pykakasi
yt-dlp
pandas
python-multipart
uvicorn
```

Install all at once:

```bash
pip install deep-translator fastapi faster-whisper argostranslate pyannote.audio jamdict jamdict-data sudachipy sudachidict-core pykakasi yt-dlp pandas python-multipart uvicorn
```

---

## Setup

### 1. Clone / copy the project

```
hayaku-pro/
├── server.py
├── static/
│   └── index.html
├── storage/
│   ├── db/
│   └── media/
│       ├── uploads/
│       └── downloads/
├── start.sh
└── core/
    ├── __init__.py     ← create this empty file
    ├── engine.py
    ├── linguistics.py
    ├── processor.py
    ├── database.py
    ├── models.py
    ├── config.py
    └── utils.py
```

> **Important:** The `core/` folder must be a Python package. Create an empty `core/__init__.py` if it doesn't exist:
> ```bash
> touch core/__init__.py
> ```

### 2. Install dictionary + translation data

```bash
pip install jamdict jamdict-data
```

Argos language package (Japanese → English) installs automatically on first translation call.

Translation provider controls:

| Variable | Default | Purpose |
|---|---|---|
| `TRANSLATE_PROVIDER` | `google_then_argos` | `google_then_argos`, `google`, or `argos` |
| `TRANSLATE_SOURCE_LANG` | `ja` | Source language code |
| `TRANSLATE_TARGET_LANG` | `en` | Target language code |

### 3. Configure speaker tracking (pyannote)

You need a Hugging Face token with access to the pyannote model:

```bash
export HF_TOKEN=your_huggingface_token
```

Optional controls:

| Variable | Default | Purpose |
|---|---|---|
| `SPEAKER_TRACKING_ENABLED` | `1` | Set to `0` to disable diarization |
| `PYANNOTE_MODEL` | `pyannote/speaker-diarization-3.1` | Diarization model id |
| `DIARIZATION_DEVICE` | `WHISPER_DEVICE` | `cuda` or `cpu` for diarization |

### 4. Configure Whisper *(optional)*

Whisper model size and device are set via environment variables. Defaults work out of the box.

| Variable | Default | Options |
|---|---|---|
| `WHISPER_MODEL` | `medium` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cuda` | `cuda` |
| `WHISPER_COMPUTE` | `float16` | `int8`, `float16`, `float32` |

Override before running:

```bash
export WHISPER_MODEL=large-v3
export WHISPER_DEVICE=cuda
```

Whisper is GPU-only in this project. If CUDA init/runtime fails, processing stops with an explicit error.

### 5. Launch

```bash
./start.sh
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Usage

### Immersion Room

The main page. Two input methods:

**YouTube link**
1. Paste any YouTube URL into the text box
2. Click **Analyze**
3. Video opens immediately and you can watch while transcription/translation runs in segment chunks
4. Transcript and subtitle cards fill in progressively as chunks complete

**Local file**
1. Drag and drop an `.mp4`, `.mp3`, or `.wav` file
2. Click **Analyze**
3. Player opens immediately; processing continues in the background with progressive transcript updates

**Using the transcript**
- **Hover any word** → tooltip shows the word, its romaji reading, and the dictionary meaning
- **Click a timestamp button** → the video player jumps to that segment
- **Speaker labels** are shown per segment (e.g. `SPEAKER_1`, `SPEAKER_2`)
- **Full-sentence English translation** is shown for the active subtitle segment
- Words are split at the morpheme level by Sudachi, so compound words and conjugations are handled correctly
- **Recent Analyses** saves locally so you can reopen prior transcripts
- **Browser back/forward** now keeps navigation accessible instead of dropping out immediately
- **Transcript tools**: live filter, copy current sentence, JSON export
- **Anki export**: one-click TSV export (`Word`, `Reading`, `Meaning`, `Sentence`, `Translation`)
- **Immersion toggles**: Furigana show/hide + subtitle blur mode for listening-first sessions
- **Theme switcher**: Noir Violet, Graphite Blue, Kuro Pink, Warm Beige
- **Player shortcuts**: `j` = previous segment, `k` = next segment, `f` = furigana toggle, `b` = blur toggle
- **Whisper runtime controls**: choose model from UI and enable RAM fallback mode when VRAM is tight

### Vocab Analytics

Every word encountered across all sessions is saved to a local SQLite database. This page shows:

- Total unique words encountered
- How many appear more than 3 times (high-frequency → high value to learn)
- A sortable table of all words with reading, meaning, and discovery count

### System Settings

- Shows your current Whisper model, device, and compute settings
- Shows the database file path
- **Clear All Vocabulary** button to reset the vocab bank

---

## How dictionary lookup works

The app tokenizes each sentence with Sudachi and looks each token up in Jamdict using the token's **dictionary form**.

Example:

```
Surface token: 食べています
Dictionary form: 食べる
Meaning returned: to eat
```

This keeps lookups instant while still handling conjugations correctly.

---

## Data storage

Everything is stored under `storage/` to keep the project root clean.

Main database file: `storage/db/immersion.db`

| Table | Contents |
|---|---|
| `vocabulary` | Every unique word: text, reading, meaning, encounter count, known flag, notes |
| `lessons` | Processed media files with metadata |
| `segments` | Transcript segments with token JSON, linked to lessons |
| `jobs` | Async processing job state |

The database uses WAL mode for better read performance. To change the path, edit `core/config.py`:

```python
class DBSettings:
    path: str = "storage/db/immersion.db"
```

---

## Troubleshooting

**"Jamdict lookup failed"** in the logs
→ Install dictionary package: `pip install jamdict jamdict-data`

**"Speaker tracking disabled"** in logs
→ Set `HF_TOKEN` with access to `pyannote/speaker-diarization-3.1`
→ Install diarization package: `pip install pyannote.audio`

**"Sentence translation unavailable"** in logs
→ Install translation package: `pip install argostranslate`
→ Ensure internet on first run so Argos can download ja→en model

**Whisper GPU initialization failed / runtime failed**
→ Ensure CUDA libraries are visible (`LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH`)
→ Keep `WHISPER_DEVICE=cuda` (CPU mode is disabled)

**YouTube download fails**
→ Update yt-dlp: `pip install -U yt-dlp` (YouTube's formats change frequently)

**`ModuleNotFoundError: No module named 'core'`**
→ Make sure you're running `./start.sh` (or `python -m uvicorn server:app`) from the project root, and that `core/__init__.py` exists.

**Sudachi dictionary not found**
→ Install the dictionary package: `pip install sudachidict-core`

---

## Tips for better immersion

- **Use `large-v3`** if you have a GPU — accuracy on fast speech and dialect improves significantly over `medium`
- **Sort vocab by Discovery Count** — words that keep appearing are the highest value to actually study
- **Hover unfamiliar particles** — function words still show dictionary-style glosses; pair with grammar references when needed
- **YouTube native player + audio-only download** means the video starts instantly; the analysis runs in parallel in the background
