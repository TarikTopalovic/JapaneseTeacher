# HAYAKU-LOCAL — Local Japanese Immersion Lab

Quick guide to run the Streamlit UI locally and configure the Whisper model.

Requirements
- Python 3.10+ and a virtualenv
- Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Running

```bash
# run with automatic GPU/CPU selection
streamlit run app.py

# or force a specific model/device
WHISPER_MODEL=small WHISPER_DEVICE=cuda streamlit run app.py
```

Notes
- If you want to download videos from YouTube, install `yt-dlp` (already in `requirements.txt`).
- If your GPU has limited VRAM (e.g., 8GB for RTX 3070), prefer `small`/`medium` models or use CPU.
- Set `HF_TOKEN` environment variable to improve HuggingFace download reliability.

Testing

```bash
pip install -U pytest
pytest -q
```
