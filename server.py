import os, uuid, threading, logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp

from core.engine import AudioEngine
from core.linguistics import LinguisticsManager
from core.sentence_translation import SentenceTranslator
from core.database import ImmersionDB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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

def _run(jid, audio_path, cleanup=False):
    try:
        _upd(jid, status='transcribing')
        raw   = engine.transcribe(audio_path)
        total = len(raw)
        _upd(jid, status='translating', total=total, progress=0)
        results, done = [None]*total, 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            fmap = {pool.submit(_process_seg, s): i for i, s in enumerate(raw)}
            for future in as_completed(fmap):
                i = fmap[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    log.error(f'Failed processing segment {i}: {e}')
                    results[i] = {
                        **raw[i],
                        'speaker': raw[i].get('speaker', 'SPEAKER_1'),
                        'translation': translator.translate(raw[i].get('text', '')),
                        'tokens': [],
                    }
                done += 1; _upd(jid, progress=done)
        if cleanup and os.path.exists(audio_path): os.remove(audio_path)
        _upd(jid, status='done', result=results)
    except Exception as e:
        _upd(jid, status='error', error=str(e))

app = FastAPI(title='Hayaku Pro')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.post('/api/analyze/youtube')
async def analyze_yt(url: str = Query(...), bg: BackgroundTasks = BackgroundTasks()):
    jid = str(uuid.uuid4())
    _jobs[jid] = {'status': 'downloading', 'progress': 0, 'total': 0, 'result': None, 'error': None}
    def go():
        try:
            opts = {'format':'bestaudio/best','outtmpl':f'/tmp/hayaku_{jid}.%(ext)s','quiet':True,'no_warnings':True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                path = ydl.prepare_filename(info)
            _run(jid, path, cleanup=True)
        except Exception as e: _upd(jid, status='error', error=str(e))
    bg.add_task(go); return {'job_id': jid}

@app.post('/api/analyze/upload')
async def analyze_upload(file: UploadFile = File(...), bg: BackgroundTasks = BackgroundTasks()):
    jid  = str(uuid.uuid4())
    path = f'/tmp/hayaku_{jid}_{file.filename}'
    with open(path, 'wb') as f: f.write(await file.read())
    _jobs[jid] = {'status': 'queued', 'progress': 0, 'total': 0, 'result': None, 'error': None}
    bg.add_task(_run, jid, path, True); return {'job_id': jid}

@app.get('/api/job/{jid}')
def get_job(jid: str):
    job = _jobs.get(jid)
    if not job: raise HTTPException(404, 'Not found')
    if job['status'] == 'done': return job
    return {k: v for k,v in job.items() if k != 'result'}

@app.get('/api/vocab')
def get_vocab():
    try:
        rows = db.get_all()
        if not rows: return []
        if isinstance(rows[0], dict): return rows
        return [{'word':r[0],'reading':r[1],'meaning':r[2],'count':r[3]} for r in rows]
    except Exception as e:
        log.error(f'vocab: {e}'); return []

static_dir = Path(__file__).parent / 'static'
static_dir.mkdir(exist_ok=True)
app.mount('/', StaticFiles(directory=str(static_dir), html=True), name='static')
