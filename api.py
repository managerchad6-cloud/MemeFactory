#!/usr/bin/env python3
"""
FastAPI server for Virgin vs Chad meme generation.
Receives JSON payload, runs generation in a background thread,
and exposes job-polling endpoints so clients never time out.
"""

import os
import json
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


# =========================
# App
# =========================

DB_PATH = Path(__file__).parent / "memes.db"


def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS memes (
            job_id       TEXT PRIMARY KEY,
            meme_id      TEXT,
            status       TEXT NOT NULL DEFAULT 'processing',
            created_at   REAL NOT NULL,
            completed_at REAL
        )
    """)
    con.commit()
    con.close()


_init_db()


_OPENAI_MODEL = "gpt-5.2"

app = FastAPI(
    title="MemeFactory API",
    description="Generate Virgin vs Chad memes via API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Models
# =========================

class MemeRequest(BaseModel):
    """STRICT request (labels required)"""
    virgin: str = Field(..., min_length=1)
    chad: str = Field(..., min_length=1)
    virgin_labels: List[str] = Field(..., min_items=1, max_items=12)
    chad_labels: List[str] = Field(..., min_items=1, max_items=12)


class MemeRequestRaw(BaseModel):
    """RAW request (labels optional)"""
    virgin: str = Field(..., min_length=1)
    chad: str = Field(..., min_length=1)
    virgin_labels: Optional[List[str]] = Field(default=None, max_items=12)
    chad_labels: Optional[List[str]] = Field(default=None, max_items=12)


class MemeRequestFreestyle(BaseModel):
    """FREESTYLE request — any natural-language text describing the two archetypes."""
    text: str = Field(..., min_length=1, max_length=500)


class MemeParseRequest(BaseModel):
    """Parse-only — extract archetypes from free-form text without generating a meme."""
    text: str = Field(..., min_length=1, max_length=500)


# =========================
# Job store
# =========================

_jobs: dict = {}
_jobs_lock = threading.Lock()


def _new_job_id() -> str:
    return f"api_{int(time.time() * 1000)}"


def _submit_job(virgin, chad, virgin_labels, chad_labels) -> str:
    job_id = _new_job_id()
    now = time.time()
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "image_path": None, "error": None}
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO memes (job_id, status, created_at) VALUES (?, 'processing', ?)",
        (job_id, now)
    )
    con.commit()
    con.close()
    threading.Thread(
        target=_run_generation_bg,
        args=(job_id, virgin, chad, virgin_labels, chad_labels),
        daemon=True,
    ).start()
    return job_id


def _run_generation_bg(job_id, virgin, chad, virgin_labels, chad_labels):
    try:
        image = _run_generation(job_id, virgin, chad, virgin_labels, chad_labels)
        ideas_file = Path(__file__).parent / "jobs" / job_id / "ideas.json"
        meme_id = None
        if ideas_file.exists():
            data = json.load(open(ideas_file))
            if data.get("items"):
                meme_id = data["items"][0].get("id")
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["image_path"] = str(image)
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "UPDATE memes SET status='done', meme_id=?, completed_at=? WHERE job_id=?",
            (meme_id, time.time(), job_id)
        )
        con.commit()
        con.close()
    except HTTPException as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = e.detail
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "UPDATE memes SET status='failed', completed_at=? WHERE job_id=?",
            (time.time(), job_id)
        )
        con.commit()
        con.close()
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "UPDATE memes SET status='failed', completed_at=? WHERE job_id=?",
            (time.time(), job_id)
        )
        con.commit()
        con.close()


# =========================
# Core generator
# =========================

def _run_generation(
    job_id: str,
    virgin: str,
    chad: str,
    virgin_labels: Optional[List[str]],
    chad_labels: Optional[List[str]],
):
    base_dir = Path(__file__).parent
    job_dir = base_dir / "jobs" / job_id
    out_dir = job_dir / "out"

    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    ideas_file = job_dir / "ideas.json"

    # Step 1: prompt generation
    env = os.environ.copy()
    env["IDEAS_FILE"] = str(ideas_file)

    payload = {
        "virgin": virgin,
        "chad": chad,
        "virgin_labels": virgin_labels or [],
        "chad_labels": chad_labels or [],
    }

    result = subprocess.run(
        ["python3", str(base_dir / "generate_idea.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(base_dir),
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Prompt generation failed: {result.stderr}"
        )

    if not ideas_file.exists():
        raise HTTPException(status_code=500, detail="ideas.json not created")

    # Step 2: image generation
    env["OUT_DIR"] = str(out_dir)

    result = subprocess.run(
        ["python3", str(base_dir / "run_batch.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(base_dir),
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {result.stderr}"
        )

    images = list(out_dir.glob("*.png"))
    if not images:
        raise HTTPException(status_code=500, detail="No output image found")

    return max(images, key=lambda p: p.stat().st_mtime)


# =========================
# Generate endpoints
# =========================

@app.post("/generate", status_code=202)
async def generate_strict(request: MemeRequest):
    """
    STRICT endpoint. Labels required.
    Returns job_id immediately; poll /jobs/{job_id} for status.
    """
    job_id = _submit_job(
        request.virgin, request.chad,
        request.virgin_labels, request.chad_labels,
    )
    return {"job_id": job_id, "status": "processing"}


@app.post("/generate/raw", status_code=202)
async def generate_raw(request: MemeRequestRaw):
    """
    RAW endpoint. Labels optional.
    Returns job_id immediately; poll /jobs/{job_id} for status.
    """
    job_id = _submit_job(
        request.virgin, request.chad,
        request.virgin_labels, request.chad_labels,
    )
    return {"job_id": job_id, "status": "processing"}


def _parse_freestyle(text: str) -> dict:
    """
    Use the LLM to extract virgin, chad, and optional labels from any natural-language input.
    Handles typos, missing keywords, any separator or label format.
    Raises HTTPException 422 if the two archetypes cannot be identified.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You parse Virgin vs Chad meme descriptions written by chaotic users.\n"
                    "\n"
                    "Your job is to extract:\n"
                    "  1. The two archetypes (required).\n"
                    "  2. Any labels/traits the user provided for each side (optional).\n"
                    "\n"
                    "INPUT RULES — the text can be anything:\n"
                    "  - May or may not include the words 'virgin'/'chad'.\n"
                    "  - Archetypes may be separated by 'vs', 'and', a comma, a slash, whitespace, or nothing.\n"
                    "  - Labels may appear in parentheses, brackets, after a colon or dash, as bullet points,\n"
                    "    inline after the name, or mixed across the line — in any order, with any punctuation.\n"
                    "  - Typos, abbreviations, and inconsistent casing are expected.\n"
                    "  - There may be zero labels, labels for only one side, or labels for both sides.\n"
                    "\n"
                    "OUTPUT — return ONLY a JSON object with these keys:\n"
                    "  \"virgin\"        : string  — clean noun phrase, Title Case, no 'Virgin' prefix\n"
                    "  \"chad\"          : string  — clean noun phrase, Title Case, no 'Chad' prefix\n"
                    "  \"virgin_labels\" : array of strings  — traits for the virgin side ([] if none found)\n"
                    "  \"chad_labels\"   : array of strings  — traits for the chad side ([] if none found)\n"
                    "\n"
                    "Label strings should be short, clean phrases as the user intended them.\n"
                    "Do NOT invent labels that weren't in the input.\n"
                    "If you cannot identify two distinct archetypes, return {\"error\": \"<reason>\"}."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)

    if "error" in data:
        raise HTTPException(status_code=422, detail=f"Could not parse archetypes: {data['error']}")

    virgin = str(data.get("virgin", "")).strip()
    chad = str(data.get("chad", "")).strip()

    if not virgin or not chad:
        raise HTTPException(status_code=422, detail="Could not extract both archetypes from text")

    virgin_labels = [str(l).strip() for l in data.get("virgin_labels", []) if str(l).strip()]
    chad_labels = [str(l).strip() for l in data.get("chad_labels", []) if str(l).strip()]

    return {
        "virgin": virgin,
        "chad": chad,
        "virgin_labels": virgin_labels,
        "chad_labels": chad_labels,
    }


@app.post("/generate/freestyle", status_code=202)
async def generate_freestyle(request: MemeRequestFreestyle):
    """
    FREESTYLE endpoint. Accepts any natural-language description of the two archetypes,
    with optional labels in any format imaginable.
    Examples:
      "virgin python dev (slow, uses pip, scared of types) vs chad rust dev (blazing fast, memory safe)"
      "tabs [rigid, old school, sane] and spaces [chaotic, smug, PEP 8 cultist]"
      "morning person night owl"
    Parses everything with an LLM, then runs the normal async generation pipeline.
    Returns job_id + what was parsed so you can verify the extraction.
    """
    parsed = _parse_freestyle(request.text)
    job_id = _submit_job(
        parsed["virgin"], parsed["chad"],
        parsed["virgin_labels"] or None,
        parsed["chad_labels"] or None,
    )
    return {"job_id": job_id, "status": "processing", "parsed": parsed}


@app.post("/parse")
async def parse_meme(request: MemeParseRequest):
    """
    Parse a natural-language meme description using LLM.
    Returns extracted virgin/chad names and labels without starting generation.
    """
    return _parse_freestyle(request.text)


# =========================
# Job endpoints
# =========================

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Poll for job status. status: processing | done | failed"""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"], "error": job["error"]}


@app.get("/jobs/{job_id}/image", response_class=FileResponse)
async def job_image(job_id: str):
    """Fetch the generated image once status is done."""
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job not ready")
    p = Path(job["image_path"])
    return FileResponse(path=str(p), media_type="image/png", filename=p.name)


@app.get("/jobs/{job_id}/metadata")
async def job_metadata(job_id: str):
    """
    Return meme metadata for a job: character names, virgin/chad labels, and meme ID.
    Available as soon as prompt generation completes (before image is ready).
    """
    ideas_file = Path(__file__).parent / "jobs" / job_id / "ideas.json"
    if not ideas_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not found for job")

    with open(ideas_file, "r") as f:
        ideas = json.load(f)

    items = ideas.get("items", [])
    if not items:
        raise HTTPException(status_code=404, detail="No meme data in job")

    item = items[0]
    return {
        "job_id": job_id,
        "id": item.get("id"),
        "virgin_labels": item.get("virgin_labels", []),
        "chad_labels": item.get("chad_labels", []),
    }


# =========================
# Meme index endpoint
# =========================

def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.get("/memes")
async def list_memes(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None, pattern="^(processing|done|failed)$"),
):
    """Paginated index of all memes generated via this API (newest first)."""
    offset = (page - 1) * limit
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        if status:
            total = con.execute(
                "SELECT COUNT(*) FROM memes WHERE status=?", (status,)
            ).fetchone()[0]
            rows = con.execute(
                "SELECT job_id, meme_id, status, created_at, completed_at "
                "FROM memes WHERE status=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset)
            ).fetchall()
        else:
            total = con.execute("SELECT COUNT(*) FROM memes").fetchone()[0]
            rows = con.execute(
                "SELECT job_id, meme_id, status, created_at, completed_at "
                "FROM memes ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
    finally:
        con.close()

    items = [
        {
            "job_id": r["job_id"],
            "meme_id": r["meme_id"],
            "status": r["status"],
            "created_at": _ts_to_iso(r["created_at"]),
            "completed_at": _ts_to_iso(r["completed_at"]),
        }
        for r in rows
    ]
    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "has_next": offset + limit < total,
        "has_prev": page > 1,
    }


# =========================
# Health
# =========================

@app.get("/")
async def root():
    return {"service": "MemeFactory API", "status": "operational", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
