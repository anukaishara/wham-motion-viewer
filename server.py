#!/usr/bin/env python3
"""
WHAM Unified Server
Serves the Motion Viewer React frontend and provides API endpoints for:
  POST /api/upload          — accept a video, start WHAM pipeline, return job_id
  GET  /api/progress/{id}   — SSE stream of pipeline progress updates
  GET  /api/results/{id}/{filename} — serve output files to the viewer
  GET  /api/jobs/{id}       — job status

Run:
  conda run -n wham_dev python server.py
  Then open: http://localhost:8787
"""

import os
import re
import sys
import json
import uuid
import time
import asyncio
import os.path as osp
from glob import glob
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ─────────────────────────────────────────────────────────────────────
WHAM_DIR     = osp.dirname(osp.abspath(__file__))
FRONTEND_DIST = osp.join(WHAM_DIR, 'motion_viewer/dist')
UPLOADS_DIR  = osp.join(WHAM_DIR, 'uploads')
OUTPUTS_DIR  = osp.join(WHAM_DIR, 'outputs')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="WHAM Unified Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry: job_id → dict
# Survives for the lifetime of this server process only.
_jobs: dict[str, dict] = {}


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # Sanitize filename: keep alphanumeric, underscores, hyphens only
    raw_name = osp.splitext(file.filename)[0]
    ext      = osp.splitext(file.filename)[1].lower()
    safe_name = re.sub(r'[^\w\-]', '_', raw_name).strip('_') or 'video'

    job_id     = str(uuid.uuid4())[:8]
    upload_path = osp.join(UPLOADS_DIR, f"{job_id}_{safe_name}{ext}")
    # pipeline.py derives video_name from the input file stem, which is "{job_id}_{safe_name}"
    video_file_stem = f"{job_id}_{safe_name}"
    output_dir  = osp.join(OUTPUTS_DIR, video_file_stem)
    os.makedirs(output_dir, exist_ok=True)

    content = await file.read()
    with open(upload_path, 'wb') as fh:
        fh.write(content)

    _jobs[job_id] = {
        'video_name':  safe_name,
        'video_path':  upload_path,
        'output_dir':  output_dir,
        'status':      'queued',
        'process':     None,
    }

    # Launch pipeline as background task — does not block the HTTP response
    asyncio.create_task(_run_pipeline(job_id))

    return {'job_id': job_id, 'video_name': safe_name}


# ── Pipeline runner ───────────────────────────────────────────────────────────

async def _run_pipeline(job_id: str):
    job           = _jobs[job_id]
    progress_path = osp.join(job['output_dir'], 'progress.json')

    _write_progress(progress_path, {
        'stage': 0, 'stage_name': 'Starting', 'pct': 1, 'status': 'running',
    })
    _jobs[job_id]['status'] = 'running'

    cmd = [
        sys.executable, osp.join(WHAM_DIR, 'pipeline.py'),
        '--input',         osp.abspath(job['video_path']),
        '--output_dir',    osp.abspath(OUTPUTS_DIR),
        '--progress_file', osp.abspath(progress_path),
    ]
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=WHAM_DIR,
        )
        _jobs[job_id]['process'] = proc

        # Drain stdout so the pipe doesn't fill up and deadlock the subprocess
        async for _ in proc.stdout:
            pass

        await proc.wait()

        if proc.returncode == 0:
            _jobs[job_id]['status'] = 'done'
            _write_progress(progress_path, {
                'stage': 4, 'stage_name': 'Complete',
                'pct': 100, 'status': 'done',
            })
        else:
            _jobs[job_id]['status'] = 'error'
            _write_progress(progress_path, {
                'status': 'error',
                'message': f'Pipeline exited with code {proc.returncode}',
            })

    except Exception as exc:
        _jobs[job_id]['status'] = 'error'
        _write_progress(progress_path, {
            'status': 'error',
            'message': str(exc),
        })


def _write_progress(path: str, data: dict):
    data['ts'] = time.time()
    try:
        with open(path, 'w') as fh:
            json.dump(data, fh)
    except Exception:
        pass


# ── SSE progress stream ───────────────────────────────────────────────────────

@app.get("/api/progress/{job_id}")
async def stream_progress(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    progress_path = osp.join(_jobs[job_id]['output_dir'], 'progress.json')

    async def _generator():
        last_json = None
        # Give the pipeline up to 60 s to start writing its first progress entry
        deadline = time.time() + 60
        while True:
            try:
                if osp.exists(progress_path):
                    with open(progress_path) as fh:
                        raw = fh.read()
                    if raw and raw != last_json:
                        last_json = raw
                        yield f"data: {raw}\n\n"
                        data = json.loads(raw)
                        if data.get('status') in ('done', 'error'):
                            return
                    deadline = time.time() + 60   # reset on any activity
            except Exception:
                pass

            if time.time() > deadline:
                yield 'data: {"status":"error","message":"Pipeline timed out (no progress)"}\n\n'
                return

            await asyncio.sleep(0.4)

    return StreamingResponse(
        _generator(),
        media_type='text/event-stream',
        headers={
            'Cache-Control':   'no-cache',
            'X-Accel-Buffering': 'no',   # disable nginx buffering if behind proxy
        },
    )


# ── Result file serving ───────────────────────────────────────────────────────

@app.get("/api/results/{job_id}/{filename}")
async def get_result(job_id: str, filename: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    out_dir = _jobs[job_id]['output_dir']

    # Direct path first
    path = osp.join(out_dir, filename)
    if osp.exists(path):
        return FileResponse(path)

    # Alias: "processed_video.mp4" → first *_processed.mp4 in the output dir
    if filename == 'processed_video.mp4':
        candidates = glob(osp.join(out_dir, '*_processed.mp4'))
        if candidates:
            return FileResponse(candidates[0])

    # Alias: "input_video.mp4" → the original uploaded file
    if filename == 'input_video.mp4':
        video_path = _jobs[job_id].get('video_path', '')
        if video_path and osp.exists(video_path):
            return FileResponse(video_path)

    raise HTTPException(status_code=404, detail=f"{filename} not found in job output")


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    j = _jobs[job_id]
    return {'job_id': job_id, 'video_name': j['video_name'], 'status': j['status']}


# ── Serve React frontend ──────────────────────────────────────────────────────
# The React app is built to motion_viewer_ipcv/dist. Mount it last so API
# routes above take precedence.
if osp.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    @app.get("/")
    async def _no_frontend():
        return HTMLResponse(
            "<h1>Frontend not built.</h1>"
            "<p>Run: <code>cd motion_viewer_ipcv && npm run build</code></p>"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("WHAM Unified Server")
    print(f"  API  → http://localhost:8787/api/")
    print(f"  App  → http://localhost:8787/")
    if not osp.isdir(FRONTEND_DIST):
        print("  WARNING: frontend not built — run: cd motion_viewer_ipcv && npm run build")
    uvicorn.run(app, host='0.0.0.0', port=8787, reload=False)
