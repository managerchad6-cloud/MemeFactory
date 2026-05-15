#!/usr/bin/env python3
"""
FastAPI server for Virgin vs Chad meme generation.
Receives JSON payload, runs generation in a background thread,
and exposes job-polling endpoints so clients never time out.

Every generation endpoint requires a confirmed Solana transaction
that paid >= 1 USDC to the treasury. The transaction signature is
consumed on first use — replay attacks are rejected at the DB layer.
"""

import os
import json
import re
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
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
# Payment constants
# =========================

_USDC_MINT     = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
_TREASURY      = "BvqPmrhAMJHozjpmJ9r7zLwkbZbS99pSaEkfQw3HxUQS"
_USDC_REQUIRED = 1_000_000  # 1 USDC — token uses 6 decimal places
_TX_MAX_AGE    = 600        # seconds — reject transactions older than 10 minutes

# Base58 character set validation (no 0/O/I/l — classic base58 alphabet)
_TX_SIG_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{87,88}$")
_ADDR_RE   = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


# =========================
# App / DB
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
    for col, typedef in [("wallet", "TEXT"), ("signature", "TEXT")]:
        try:
            con.execute(f"ALTER TABLE memes ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass  # column already exists

    old_schema = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='votes'"
    ).fetchone()
    needs_migration = old_schema and "PRIMARY KEY (job_id, wallet)" not in old_schema[0]
    if needs_migration:
        con.execute("ALTER TABLE votes RENAME TO votes_old")
        con.execute("""
            CREATE TABLE votes (
                job_id     TEXT NOT NULL,
                wallet     TEXT NOT NULL,
                signature  TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY (job_id, wallet)
            )
        """)
        con.execute(
            "INSERT OR IGNORE INTO votes (job_id, wallet, signature, created_at) "
            "SELECT job_id, wallet, signature, MIN(created_at) "
            "FROM votes_old GROUP BY job_id, wallet"
        )
        con.execute("DROP TABLE votes_old")
    else:
        con.execute("""
            CREATE TABLE IF NOT EXISTS votes (
                job_id     TEXT NOT NULL,
                wallet     TEXT NOT NULL,
                signature  TEXT,
                created_at REAL NOT NULL,
                PRIMARY KEY (job_id, wallet)
            )
        """)

    # Single-use transaction registry — prevents replay attacks
    con.execute("""
        CREATE TABLE IF NOT EXISTS used_transactions (
            tx_signature TEXT PRIMARY KEY,
            wallet       TEXT NOT NULL,
            used_at      REAL NOT NULL
        )
    """)
    con.commit()
    con.close()


_init_db()


# =========================
# Payment verification
# =========================

def _rpc_call(url: str, payload: dict, timeout: float = 15.0) -> dict:
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"RPC HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"RPC connection error: {e.reason}")
    except TimeoutError:
        raise RuntimeError("RPC request timed out — try again")


def verify_usdc_payment(tx_signature: str, wallet: str) -> tuple[bool, str]:
    """
    Verify that tx_signature is a real, confirmed, unspent Solana transaction
    that transferred >= 1 USDC to the treasury from the claimed wallet.

    Checks (in order):
      1. Format validation — reject garbage before touching the network
      2. Replay protection — tx must not already be in used_transactions
      3. On-chain existence and confirmation via Helius RPC
      4. Transaction succeeded (meta.err is null)
      5. Freshness — block time within TX_MAX_AGE seconds
      6. USDC amount — treasury balance delta >= USDC_REQUIRED
      7. Signer match — wallet must appear in the tx signers list
      8. Atomic consumption — INSERT with UNIQUE guard prevents TOCTOU races

    Returns (True, "") on success, (False, reason) on any failure.
    """
    if not _TX_SIG_RE.match(tx_signature):
        return False, "Invalid transaction signature format"
    if not _ADDR_RE.match(wallet):
        return False, "Invalid wallet address format"

    # Check replay before hitting the network
    con = sqlite3.connect(DB_PATH)
    try:
        already_used = con.execute(
            "SELECT 1 FROM used_transactions WHERE tx_signature = ?", (tx_signature,)
        ).fetchone()
    finally:
        con.close()
    if already_used:
        return False, "Transaction already used for a generation"

    rpc_url = os.environ.get("HELIUS_RPC_URL", "").strip()
    if not rpc_url:
        return False, "Payment verification unavailable (HELIUS_RPC_URL not configured)"

    try:
        data = _rpc_call(rpc_url, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                tx_signature,
                {
                    "encoding": "jsonParsed",
                    "commitment": "confirmed",
                    "maxSupportedTransactionVersion": 0,
                },
            ],
        })
    except RuntimeError as e:
        return False, f"Payment verification failed: {e}"

    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        return False, f"RPC error: {msg}"

    result = data.get("result")
    if not result:
        return False, "Transaction not found — wait for confirmation and retry"

    # Transaction must have succeeded on-chain
    if result.get("meta", {}).get("err") is not None:
        return False, "Transaction failed on-chain"

    # Reject pre-saved or replayed transactions
    block_time = result.get("blockTime")
    if not block_time:
        return False, "Cannot determine transaction timestamp"
    age = time.time() - block_time
    if age > _TX_MAX_AGE:
        return False, f"Transaction is {int(age)}s old (limit is {_TX_MAX_AGE}s)"

    # Verify USDC reached the treasury
    meta          = result.get("meta", {})
    pre_balances  = meta.get("preTokenBalances")  or []
    post_balances = meta.get("postTokenBalances") or []

    pre_by_idx = {
        b["accountIndex"]: int(b["uiTokenAmount"]["amount"])
        for b in pre_balances
        if b.get("mint") == _USDC_MINT
    }
    treasury_delta = 0
    for b in post_balances:
        if b.get("mint")  != _USDC_MINT:
            continue
        if b.get("owner") != _TREASURY:
            continue
        post_amt = int(b["uiTokenAmount"]["amount"])
        pre_amt  = pre_by_idx.get(b["accountIndex"], 0)
        delta    = post_amt - pre_amt
        if delta > treasury_delta:
            treasury_delta = delta

    if treasury_delta < _USDC_REQUIRED:
        paid = treasury_delta / 1_000_000
        need = _USDC_REQUIRED / 1_000_000
        return False, f"Insufficient payment: {paid:.2f} USDC received, {need:.2f} required"

    # Confirm the transaction was signed by the wallet the caller claims
    account_keys = (
        result.get("transaction", {})
              .get("message", {})
              .get("accountKeys") or []
    )
    signers = [
        k["pubkey"] for k in account_keys
        if isinstance(k, dict) and k.get("signer")
    ]
    if signers and wallet not in signers:
        return False, "Transaction was not signed by the claimed wallet"

    # Atomically consume the transaction — UNIQUE constraint is the final
    # guard against concurrent requests racing through the check above.
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO used_transactions (tx_signature, wallet, used_at) VALUES (?, ?, ?)",
            (tx_signature, wallet, time.time()),
        )
        con.commit()
        con.close()
    except sqlite3.IntegrityError:
        return False, "Transaction already used for a generation"

    return True, ""


# =========================
# Models
# =========================

class MemeRequest(BaseModel):
    """STRICT endpoint — labels required on both sides."""
    virgin:        str       = Field(..., min_length=1)
    chad:          str       = Field(..., min_length=1)
    virgin_labels: List[str] = Field(..., min_items=1, max_items=12)
    chad_labels:   List[str] = Field(..., min_items=1, max_items=12)
    tx_signature:  str       = Field(..., min_length=1)
    wallet:        str       = Field(..., min_length=1)


class MemeRequestRaw(BaseModel):
    """RAW endpoint — labels optional."""
    virgin:        str                  = Field(..., min_length=1)
    chad:          str                  = Field(..., min_length=1)
    virgin_labels: Optional[List[str]]  = Field(default=None, max_items=12)
    chad_labels:   Optional[List[str]]  = Field(default=None, max_items=12)
    tx_signature:  str                  = Field(..., min_length=1)
    wallet:        str                  = Field(..., min_length=1)


class MemeRequestFreestyle(BaseModel):
    """FREESTYLE endpoint — natural-language input, LLM parses it first."""
    text:         str = Field(..., min_length=1, max_length=500)
    tx_signature: str = Field(..., min_length=1)
    wallet:       str = Field(..., min_length=1)


class MemeParseRequest(BaseModel):
    """Parse-only — no generation, no payment required."""
    text: str = Field(..., min_length=1, max_length=500)


# =========================
# Job store
# =========================

_jobs: dict = {}
_jobs_lock = threading.Lock()

_OPENAI_MODEL = "gpt-5.2"


def _new_job_id() -> str:
    return f"api_{int(time.time() * 1000)}"


def _submit_job(virgin, chad, virgin_labels, chad_labels, wallet=None, tx_signature=None) -> str:
    job_id = _new_job_id()
    now = time.time()
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "image_path": None, "error": None, "wallet": wallet}
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO memes (job_id, status, created_at, wallet, signature) VALUES (?, 'processing', ?, ?, ?)",
        (job_id, now, wallet, tx_signature)
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
    job_dir  = base_dir / "jobs" / job_id
    out_dir  = job_dir  / "out"

    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    ideas_file = job_dir / "ideas.json"

    env = os.environ.copy()
    env["IDEAS_FILE"] = str(ideas_file)

    payload = {
        "virgin":        virgin,
        "chad":          chad,
        "virgin_labels": virgin_labels or [],
        "chad_labels":   chad_labels   or [],
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
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {result.stderr}")
    if not ideas_file.exists():
        raise HTTPException(status_code=500, detail="ideas.json not created")

    env["OUT_DIR"] = str(out_dir)

    result = subprocess.run(
        ["python3", str(base_dir / "run_batch.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(base_dir),
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {result.stderr}")

    images = list(out_dir.glob("*.png"))
    if not images:
        # run_batch.py exited 0 but produced no image — check its stdout/stderr
        # and the most recent log file for the actual Gemini finish_reason.
        detail = "No output image produced"
        stderr_hint = (result.stderr or "").strip()
        stdout_hint = (result.stdout or "").strip()
        hint = stderr_hint or stdout_hint
        if hint:
            # pull the first WARNING/ERROR line that mentions finish_reason or blocked
            for line in hint.splitlines():
                if "finish_reason" in line or "blocked" in line or "SAFETY" in line or "FAILED" in line:
                    detail = f"Generation blocked: {line.strip()}"
                    break
            else:
                detail = f"No output image produced. Generator output: {hint[:300]}"
        raise HTTPException(status_code=500, detail=detail)

    return max(images, key=lambda p: p.stat().st_mtime)


# =========================
# App
# =========================

app = FastAPI(
    title="MemeFactory API",
    description="Generate Virgin vs Chad memes via API",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Generate endpoints
# =========================

@app.post("/generate", status_code=202)
async def generate_strict(request: MemeRequest):
    """STRICT — labels required. Payment verified before generation starts."""
    ok, err = verify_usdc_payment(request.tx_signature, request.wallet)
    if not ok:
        raise HTTPException(status_code=402, detail=err)
    job_id = _submit_job(
        request.virgin, request.chad,
        request.virgin_labels, request.chad_labels,
        request.wallet, request.tx_signature,
    )
    return {"job_id": job_id, "status": "processing"}


@app.post("/generate/raw", status_code=202)
async def generate_raw(request: MemeRequestRaw):
    """RAW — labels optional. Payment verified before generation starts."""
    ok, err = verify_usdc_payment(request.tx_signature, request.wallet)
    if not ok:
        raise HTTPException(status_code=402, detail=err)
    job_id = _submit_job(
        request.virgin, request.chad,
        request.virgin_labels, request.chad_labels,
        request.wallet, request.tx_signature,
    )
    return {"job_id": job_id, "status": "processing"}


def _parse_freestyle(text: str) -> dict:
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
    chad   = str(data.get("chad",   "")).strip()
    if not virgin or not chad:
        raise HTTPException(status_code=422, detail="Could not extract both archetypes from text")
    return {
        "virgin":        virgin,
        "chad":          chad,
        "virgin_labels": [str(l).strip() for l in data.get("virgin_labels", []) if str(l).strip()],
        "chad_labels":   [str(l).strip() for l in data.get("chad_labels",   []) if str(l).strip()],
    }


@app.post("/generate/freestyle", status_code=202)
async def generate_freestyle(request: MemeRequestFreestyle):
    """FREESTYLE — natural-language input. Payment verified before generation starts."""
    ok, err = verify_usdc_payment(request.tx_signature, request.wallet)
    if not ok:
        raise HTTPException(status_code=402, detail=err)
    parsed = _parse_freestyle(request.text)
    job_id = _submit_job(
        parsed["virgin"], parsed["chad"],
        parsed["virgin_labels"] or None,
        parsed["chad_labels"]   or None,
        request.wallet, request.tx_signature,
    )
    return {"job_id": job_id, "status": "processing", "parsed": parsed}


@app.post("/parse")
async def parse_meme(request: MemeParseRequest):
    """Parse natural-language text into archetypes. No payment required."""
    return _parse_freestyle(request.text)


# =========================
# Job endpoints
# =========================

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    """Poll for job status: processing | done | failed"""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job:
        return {"job_id": job_id, "status": job["status"], "error": job["error"], "wallet": job.get("wallet")}

    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        "SELECT status, wallet FROM memes WHERE job_id=?", (job_id,)
    ).fetchone()
    con.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": row[0], "error": None, "wallet": row[1]}


class VoteRequest(BaseModel):
    wallet:    str           = Field(..., min_length=1)
    signature: Optional[str] = None


@app.post("/jobs/{job_id}/vote")
async def vote_meme(job_id: str, request: VoteRequest):
    """Cast a vote. One vote per wallet per meme; duplicates are a silent no-op."""
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("SELECT 1 FROM memes WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        cur = con.execute(
            "INSERT OR IGNORE INTO votes (job_id, wallet, signature, created_at) VALUES (?, ?, ?, ?)",
            (job_id, request.wallet, request.signature, time.time()),
        )
        con.commit()
        already_voted = cur.rowcount == 0
        vote_count = con.execute(
            "SELECT COUNT(*) FROM votes WHERE job_id=?", (job_id,)
        ).fetchone()[0]
    finally:
        con.close()
    return {"job_id": job_id, "vote_count": vote_count, "already_voted": already_voted}


@app.get("/jobs/{job_id}/image", response_class=FileResponse)
async def job_image(job_id: str):
    """Fetch the generated image once status is done."""
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))

    if not job:
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT status FROM memes WHERE job_id=?", (job_id,)).fetchone()
        con.close()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if row[0] != "done":
            raise HTTPException(status_code=202, detail="Job not ready")
        out_dir = Path(__file__).parent / "jobs" / job_id / "out"
        images  = list(out_dir.glob("*.png")) if out_dir.exists() else []
        if not images:
            raise HTTPException(status_code=404, detail="Image not found")
        p = max(images, key=lambda img: img.stat().st_mtime)
        return FileResponse(path=str(p), media_type="image/png", filename=p.name)

    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job not ready")
    p = Path(job["image_path"])
    return FileResponse(path=str(p), media_type="image/png", filename=p.name)


@app.get("/jobs/{job_id}/metadata")
async def job_metadata(job_id: str):
    """Return meme metadata (labels, ID). Available before the image is ready."""
    ideas_file = Path(__file__).parent / "jobs" / job_id / "ideas.json"
    if not ideas_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not found for job")
    with open(ideas_file) as f:
        ideas = json.load(f)
    items = ideas.get("items", [])
    if not items:
        raise HTTPException(status_code=404, detail="No meme data in job")
    item = items[0]
    return {
        "job_id":        job_id,
        "id":            item.get("id"),
        "virgin_labels": item.get("virgin_labels", []),
        "chad_labels":   item.get("chad_labels",   []),
    }


# =========================
# Meme index
# =========================

def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.get("/memes")
async def list_memes(
    page:   int           = Query(default=1,    ge=1),
    limit:  int           = Query(default=20,   ge=1, le=100),
    status: Optional[str] = Query(default=None, pattern="^(processing|done|failed)$"),
):
    """Paginated index of all memes (newest first)."""
    offset   = (page - 1) * limit
    vote_join = (
        "LEFT JOIN (SELECT job_id, COUNT(*) AS vote_count FROM votes GROUP BY job_id) v "
        "ON memes.job_id = v.job_id "
    )
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        if status:
            total = con.execute("SELECT COUNT(*) FROM memes WHERE status=?", (status,)).fetchone()[0]
            rows  = con.execute(
                "SELECT memes.job_id, meme_id, status, memes.created_at, completed_at, memes.wallet, "
                "COALESCE(v.vote_count, 0) AS vote_count "
                "FROM memes " + vote_join +
                "WHERE status=? ORDER BY memes.created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset)
            ).fetchall()
        else:
            total = con.execute("SELECT COUNT(*) FROM memes").fetchone()[0]
            rows  = con.execute(
                "SELECT memes.job_id, meme_id, status, memes.created_at, completed_at, memes.wallet, "
                "COALESCE(v.vote_count, 0) AS vote_count "
                "FROM memes " + vote_join +
                "ORDER BY memes.created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
    finally:
        con.close()

    return {
        "items": [
            {
                "job_id":       r["job_id"],
                "meme_id":      r["meme_id"],
                "status":       r["status"],
                "created_at":   _ts_to_iso(r["created_at"]),
                "completed_at": _ts_to_iso(r["completed_at"]),
                "wallet":       r["wallet"],
                "vote_count":   r["vote_count"],
            }
            for r in rows
        ],
        "total":    total,
        "page":     page,
        "limit":    limit,
        "has_next": offset + limit < total,
        "has_prev": page > 1,
    }


# =========================
# Leaderboard
# =========================

@app.get("/leaderboard")
async def leaderboard(limit: int = Query(default=15, ge=1, le=50)):
    """Top voted memes (done only), ordered by votes desc then oldest first."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT memes.job_id, meme_id, memes.wallet, memes.created_at, "
            "COALESCE(v.vote_count, 0) AS vote_count "
            "FROM memes "
            "LEFT JOIN (SELECT job_id, COUNT(*) AS vote_count FROM votes GROUP BY job_id) v "
            "ON memes.job_id = v.job_id "
            "WHERE memes.wallet IS NOT NULL AND memes.status = 'done' AND v.vote_count > 0 "
            "ORDER BY vote_count DESC, memes.created_at ASC "
            "LIMIT ?",
            (limit,)
        ).fetchall()
    finally:
        con.close()

    return {
        "items": [
            {
                "job_id":     r["job_id"],
                "meme_id":    r["meme_id"],
                "wallet":     r["wallet"],
                "vote_count": r["vote_count"],
                "created_at": _ts_to_iso(r["created_at"]),
            }
            for r in rows
        ]
    }


# =========================
# Health
# =========================

@app.get("/")
async def root():
    return {"service": "MemeFactory API", "status": "operational", "version": "2.1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
