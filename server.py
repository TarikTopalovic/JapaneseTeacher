import os
import uuid
import threading
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import yt_dlp

from core.engine import AudioEngine
from core.linguistics import LinguisticsManager
from core.sentence_translation import SentenceTranslator
from core.database import ImmersionDB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent
STORAGE_DIR = ROOT_DIR / 'storage'
MEDIA_DOWNLOAD_DIR = STORAGE_DIR / 'media' / 'downloads'
MEDIA_UPLOAD_DIR = STORAGE_DIR / 'media' / 'uploads'
for directory in (MEDIA_DOWNLOAD_DIR, MEDIA_UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEGMENT_CHUNK_SIZE = max(1, int(os.getenv('SEGMENT_CHUNK_SIZE', '36')))

engine = AudioEngine()
ling   = LinguisticsManager()
translator = SentenceTranslator()
db     = ImmersionDB()

_jobs: dict = {}
_lock = threading.Lock()

def _upd(jid, **kw):
    with _lock:
        if jid in _jobs: _jobs[jid].update(kw)

def _process_seg(seg):
    tokens = ling.analyze(seg['text'])
    translation = translator.translate(seg['text'])
    for t in tokens:
        try: db.add_word(t['text'], t['reading'], t['meaning'])
        except Exception as e:
            log.warning(f"Failed to persist vocab '{t.get('text', '')}': {e}")
    return {
        'start': seg['start'],
        'end': seg['end'],
        'text': seg['text'],
        'speaker': seg.get('speaker', 'SPEAKER_1'),
        'translation': translation,
        'tokens': tokens,
    }


class WhisperSettingsUpdate(BaseModel):
    model_name: str | None = None
    compute_type: str | None = None
    allow_cpu_fallback: bool | None = None


def _run(jid, audio_path, cleanup=False):
    try:
        _upd(jid, status='transcribing')
        raw   = engine.transcribe(audio_path)
        total = len(raw)
        _upd(jid, status='translating', total=total, progress=0, segments=[])
        results = [None] * total
        for chunk_start in range(0, total, SEGMENT_CHUNK_SIZE):
            chunk = raw[chunk_start:chunk_start + SEGMENT_CHUNK_SIZE]
            chunk_results = [None] * len(chunk)
            with ThreadPoolExecutor(max_workers=4) as pool:
                fmap = {pool.submit(_process_seg, s): i for i, s in enumerate(chunk)}
                for future in as_completed(fmap):
                    idx = fmap[future]
                    absolute_idx = chunk_start + idx
                    try:
                        chunk_results[idx] = future.result()
                    except Exception as e:
                        log.error(f'Failed processing segment {absolute_idx}: {e}')
                        source = raw[absolute_idx]
                        chunk_results[idx] = {
                            **source,
                            'speaker': source.get('speaker', 'SPEAKER_1'),
                            'translation': translator.translate(source.get('text', '')),
                            'tokens': [],
                        }
            for idx, seg in enumerate(chunk_results):
                results[chunk_start + idx] = seg
            done = chunk_start + len(chunk_results)
            _upd(jid, progress=done, segments=results[:done])
        if cleanup and os.path.exists(audio_path): os.remove(audio_path)
        _upd(jid, status='done', result=results, segments=results)
    except Exception as e:
        _upd(jid, status='error', error=str(e))

app = FastAPI(title='Hayaku Pro')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.post('/api/analyze/youtube')
async def analyze_yt(url: str = Query(...), bg: BackgroundTasks = BackgroundTasks()):
    jid = str(uuid.uuid4())
    _jobs[jid] = {'status': 'downloading', 'progress': 0, 'total': 0, 'result': None, 'segments': [], 'error': None}
    def go():
        try:
            outtmpl = MEDIA_DOWNLOAD_DIR / f'hayaku_{jid}.%(ext)s'
            opts = {'format': 'bestaudio/best', 'outtmpl': str(outtmpl), 'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                path = ydl.prepare_filename(info)
            _run(jid, path, cleanup=True)
        except Exception as e: _upd(jid, status='error', error=str(e))
    bg.add_task(go); return {'job_id': jid}

@app.post('/api/analyze/upload')
async def analyze_upload(file: UploadFile = File(...), bg: BackgroundTasks = BackgroundTasks()):
    jid  = str(uuid.uuid4())
    safe_name = Path(file.filename or 'upload.bin').name
    path = MEDIA_UPLOAD_DIR / f'hayaku_{jid}_{safe_name}'
    with open(path, 'wb') as f: f.write(await file.read())
    _jobs[jid] = {'status': 'queued', 'progress': 0, 'total': 0, 'result': None, 'segments': [], 'error': None}
    bg.add_task(_run, jid, str(path), True); return {'job_id': jid}

@app.get('/api/job/{jid}')
def get_job(jid: str):
    job = _jobs.get(jid)
    if not job: raise HTTPException(404, 'Not found')
    if job['status'] == 'done': return job
    return {k: v for k,v in job.items() if k != 'result'}


@app.get('/api/settings/whisper')
def get_whisper_settings():
    return engine.runtime_settings()


@app.post('/api/settings/whisper')
def update_whisper_settings(payload: WhisperSettingsUpdate):
    if payload.model_name and payload.model_name not in engine.AVAILABLE_MODELS:
        raise HTTPException(400, f'Unsupported model "{payload.model_name}"')
    if payload.compute_type and payload.compute_type not in {'int8', 'int8_float16', 'float16', 'float32'}:
        raise HTTPException(400, f'Unsupported compute_type "{payload.compute_type}"')
    engine.configure(
        model_name=payload.model_name,
        compute_type=payload.compute_type,
        allow_cpu_fallback=payload.allow_cpu_fallback,
    )
    return engine.runtime_settings()


@app.get('/api/vocab')
def get_vocab():
    try:
        rows = db.get_all()
        if not rows: return []
        if isinstance(rows[0], dict): return rows
        return [{'word':r[0],'reading':r[1],'meaning':r[2],'count':r[3]} for r in rows]
    except Exception as e:
        log.error(f'vocab: {e}'); return []

static_dir = ROOT_DIR / 'static'
static_dir.mkdir(exist_ok=True)
app.mount('/', StaticFiles(directory=str(static_dir), html=True), name='static')
