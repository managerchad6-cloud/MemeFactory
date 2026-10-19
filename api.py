#!/usr/bin/env python3
"""
FastAPI server for Virgin vs Chad meme generation.
Receives JSON payload, runs generation in a background thread,
and exposes job-polling endpoints so clients never time out.

Who is calling comes from the VVC identity session (vvc_identity, verified offline); wallet
fields in requests are ignored. Reads are public; generating, parsing and voting need a member.

Every generation is paid, either by a confirmed Solana transaction signed by the member's wallet
that sent >= 1 USDC to the treasury, or by a single-use credit token from memestudio-backend
bound to the member's wallet. Both are consumed on first use (replays rejected at the DB layer);
USDC transactions are also claimed in memestudio's suite-wide ledger so they can't pay twice.
"""

import hashlib
import sys
import hmac
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
from typing import Annotated, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
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

# Credit token shared secret — must match the Studio backend. No default: a public default would
# let anyone mint tokens. Unset = paying with credits is off.
_CREDIT_SHARED_SECRET = os.environ.get("CREDIT_SHARED_SECRET") or None
_CREDIT_TOKEN_MAX_AGE_MS = 5 * 60 * 1000

# Suite-wide used-tx ledger in memestudio-backend: one USDC tx buys one thing across all services
# (a Studio credit purchase can't also pay for a generation here). Unset secret = USDC payments off.
_MEMOSTUDIO_INTERNAL_URL = (os.environ.get("MEMOSTUDIO_INTERNAL_URL") or "http://127.0.0.1:3016").rstrip("/")
_VVC_INTERNAL_SECRET     = os.environ.get("VVC_INTERNAL_SECRET") or None
_TX_CLAIM_PURPOSE        = "factory_generate"

# /parse is free but each call is a Gemini request: cap it per wallet
_PARSE_WINDOW_SECS = 600
_PARSE_WINDOW_MAX  = 20
_PARSE_DAY_MAX     = 200


# =========================
# App / DB
# =========================

DB_PATH = Path(os.environ.get("MF_DB_PATH") or Path(__file__).parent / "memes.db")


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
    # Single-use credit tokens (memestudio already charged the credits; spend each token once)
    con.execute("""
        CREATE TABLE IF NOT EXISTS used_credit_tokens (
            sig      TEXT PRIMARY KEY,
            wallet   TEXT NOT NULL,
            used_at  REAL NOT NULL
        )
    """)
    # /parse calls per wallet per UTC day (the 10-minute window is kept in memory)
    con.execute("""
        CREATE TABLE IF NOT EXISTS parse_usage (
            wallet TEXT NOT NULL,
            day    TEXT NOT NULL,
            count  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (wallet, day)
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
# Credit token verification
# =========================

def verify_credit_token(token: "CreditToken", wallet: str) -> tuple[bool, str]:
    """Valid, fresh, signed by memestudio, for this wallet, and never used before."""
    if not _CREDIT_SHARED_SECRET:
        return False, "Paying with credits is unavailable right now"
    if token.wallet != wallet:
        return False, "Credit token belongs to another wallet"
    now_ms = int(time.time() * 1000)
    if token.ts > now_ms + 60_000 or now_ms - token.ts > _CREDIT_TOKEN_MAX_AGE_MS:
        return False, "Credit token expired — try again"
    expected = hmac.new(
        _CREDIT_SHARED_SECRET.encode(),
        f"{token.wallet}|{token.ts}|meme_generate".encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, token.sig):
        return False, "Invalid credit token"
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("INSERT INTO used_credit_tokens (sig, wallet, used_at) VALUES (?, ?, ?)",
                    (token.sig, wallet, time.time()))
        con.commit()
    except sqlite3.IntegrityError:
        return False, "Credit token already used"
    finally:
        con.close()
    return True, ""


# =========================
# Suite tx ledger
# =========================

def _post_tx_claim(body: dict) -> int:
    """POST to memestudio's ledger and return the HTTP status. Raises RuntimeError if unreachable."""
    req = urllib.request.Request(
        f"{_MEMOSTUDIO_INTERNAL_URL}/internal/tx/claim",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-vvc-internal": _VVC_INTERNAL_SECRET or ""},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(str(e))


def _release_local_tx(tx_signature: str):
    """Undo verify_usdc_payment's local consumption so the member can retry the same tx."""
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("DELETE FROM used_transactions WHERE tx_signature = ?", (tx_signature,))
        con.commit()
    finally:
        con.close()


def claim_shared_tx(tx_signature: str, wallet: str):
    """Claim the tx in the suite ledger. Fails closed: no ledger answer, no generation."""
    try:
        status = _post_tx_claim({"tx_signature": tx_signature, "wallet": wallet, "purpose": _TX_CLAIM_PURPOSE})
    except RuntimeError:
        status = None
    if status == 200:
        return
    if status == 409:
        raise HTTPException(status_code=402, detail="Transaction already used")
    _release_local_tx(tx_signature)
    raise HTTPException(status_code=503, detail="Payment verification unavailable — try again shortly")


def take_payment(wallet: str, tx_signature: Optional[str], credit_token: Optional["CreditToken"]) -> str:
    """Consume the member's payment (credits or USDC). Returns a reference stored with the job."""
    if credit_token is not None:
        ok, err = verify_credit_token(credit_token, wallet)
        ref = f"credit:{credit_token.sig}"
    elif tx_signature:
        if not _VVC_INTERNAL_SECRET:
            raise HTTPException(status_code=503, detail="Paying with USDC is unavailable right now")
        ok, err = verify_usdc_payment(tx_signature, wallet)
        ref = tx_signature
        if ok:
            claim_shared_tx(tx_signature, wallet)
    else:
        ok, err, ref = False, "Payment required: credits or a USDC transaction", ""
    if not ok:
        raise HTTPException(status_code=402, detail=err)
    return ref


# =========================
# /parse limits
# =========================

_parse_recent: dict = {}  # wallet -> call times in the current window
_parse_lock = threading.Lock()


def _take_parse_slot(wallet: str):
    """One /parse call for this wallet, or 429 once the 10-minute or daily cap is reached."""
    now = time.time()
    with _parse_lock:
        recent = [t for t in _parse_recent.get(wallet, []) if now - t < _PARSE_WINDOW_SECS]
        if len(recent) >= _PARSE_WINDOW_MAX:
            _parse_recent[wallet] = recent
            retry = int(_PARSE_WINDOW_SECS - (now - recent[0])) + 1
            raise HTTPException(status_code=429, detail="Too many parse requests — slow down",
                                headers={"Retry-After": str(retry)})
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        con = sqlite3.connect(DB_PATH)
        try:
            con.execute("INSERT OR IGNORE INTO parse_usage (wallet, day, count) VALUES (?, ?, 0)", (wallet, day))
            cur = con.execute(
                "UPDATE parse_usage SET count = count + 1 WHERE wallet = ? AND day = ? AND count < ?",
                (wallet, day, _PARSE_DAY_MAX),
            )
            con.commit()
        finally:
            con.close()
        if cur.rowcount == 0:
            raise HTTPException(status_code=429, detail="Daily parse limit reached — try again tomorrow")
        recent.append(now)
        _parse_recent[wallet] = recent


# =========================
# Models
# =========================

class CreditToken(BaseModel):
    """Short-lived HMAC token issued by the Studio backend after deducting credits."""
    wallet: str
    ts:     int
    sig:    str


Label = Annotated[str, Field(max_length=120)]


class MemeRequest(BaseModel):
    """STRICT endpoint — labels required on both sides."""
    virgin:        str         = Field(..., min_length=1, max_length=100)
    chad:          str         = Field(..., min_length=1, max_length=100)
    virgin_labels: List[Label] = Field(..., min_items=1, max_items=12)
    chad_labels:   List[Label] = Field(..., min_items=1, max_items=12)
    tx_signature:  Optional[str]         = None
    credit_token:  Optional[CreditToken] = None


class MemeRequestRaw(BaseModel):
    """RAW endpoint — labels optional. Accepts credit_token in lieu of tx_signature."""
    virgin:        str                   = Field(..., min_length=1, max_length=100)
    chad:          str                   = Field(..., min_length=1, max_length=100)
    virgin_labels: Optional[List[Label]] = Field(default=None, max_items=12)
    chad_labels:   Optional[List[Label]] = Field(default=None, max_items=12)
    tx_signature:  Optional[str]         = Field(default=None, min_length=1)
    credit_token:  Optional[CreditToken] = None


class MemeRequestFreestyle(BaseModel):
    """FREESTYLE endpoint — natural-language input. Accepts credit_token in lieu of tx_signature."""
    text:         str                   = Field(..., min_length=1, max_length=500)
    tx_signature: Optional[str]         = Field(default=None, min_length=1)
    credit_token: Optional[CreditToken] = None


class MemeParseRequest(BaseModel):
    """Parse-only — no generation, no payment required."""
    text: str = Field(..., min_length=1, max_length=500)


# =========================
# Job store
# =========================

_jobs: dict = {}
_jobs_lock = threading.Lock()

_GEMINI_TEXT_MODEL = "gemini-3.8-flash"


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

# The suite reaches this API same-origin through Apache, so no CORS by default. Tools on other
# origins (e.g. ui/ on its own port) can be allowed explicitly: MF_CORS_ORIGINS=http://localhost:8001
_cors = [o.strip() for o in os.environ.get("MF_CORS_ORIGINS", "").split(",") if o.strip()]
if _cors:
    app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["*"], allow_headers=["*"])

# The identity client lives in the identity repo (client/python). Local dev: the sibling checkout.
sys.path.insert(0, os.environ.get("VVC_IDENTITY_CLIENT_DIR")
                or str(Path(__file__).resolve().parent.parent / "identity" / "client" / "python"))
from vvc_identity import IdentityClient  # noqa: E402

identity = IdentityClient(
    public_key_path=os.environ.get("IDENTITY_PUBLIC_KEY_PATH")
    or Path(__file__).resolve().parent.parent / "identity" / ".dev-keys" / "identity_ed25519.pub.pem",
    allowed_origins=[o.strip() for o in os.environ.get("IDENTITY_ALLOWED_ORIGINS", "https://suite.virginvschad.vip").split(",") if o.strip()],
)
identity.install(app)


# =========================
# Generate endpoints
# =========================

@app.post("/generate", status_code=202)
async def generate_strict(request: MemeRequest, wallet: str = Depends(identity.member)):
    """STRICT — labels required. Paid by the member before generation starts."""
    payment = take_payment(wallet, request.tx_signature, request.credit_token)
    job_id = _submit_job(
        request.virgin, request.chad,
        request.virgin_labels, request.chad_labels,
        wallet, payment,
    )
    return {"job_id": job_id, "status": "processing"}


@app.post("/generate/raw", status_code=202)
async def generate_raw(request: MemeRequestRaw, wallet: str = Depends(identity.member)):
    """RAW — labels optional. Paid by the member before generation starts."""
    payment = take_payment(wallet, request.tx_signature, request.credit_token)
    job_id = _submit_job(
        request.virgin, request.chad,
        request.virgin_labels, request.chad_labels,
        wallet, payment,
    )
    return {"job_id": job_id, "status": "processing"}


def _parse_freestyle(text: str) -> dict:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    system_instruction = (
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
    )
    resp = client.models.generate_content(
        model=_GEMINI_TEXT_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
        ),
    )
    data = json.loads(resp.text)
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
async def generate_freestyle(request: MemeRequestFreestyle, wallet: str = Depends(identity.member)):
    """FREESTYLE — natural-language input. Parsed first, so text we can't use costs nothing."""
    parsed = _parse_freestyle(request.text)
    payment = take_payment(wallet, request.tx_signature, request.credit_token)
    job_id = _submit_job(
        parsed["virgin"], parsed["chad"],
        parsed["virgin_labels"] or None,
        parsed["chad_labels"]   or None,
        wallet, payment,
    )
    return {"job_id": job_id, "status": "processing", "parsed": parsed}


@app.post("/parse")
async def parse_meme(request: MemeParseRequest, wallet: str = Depends(identity.member)):
    """Parse natural-language text into archetypes. Free, but members only and rate-limited (it calls a paid LLM)."""
    _take_parse_slot(wallet)
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


@app.post("/jobs/{job_id}/vote")
async def vote_meme(job_id: str, wallet: str = Depends(identity.member)):
    """Cast a vote as the member. One vote per wallet per meme; duplicates are a silent no-op."""
    con = sqlite3.connect(DB_PATH)
    try:
        row = con.execute("SELECT 1 FROM memes WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        cur = con.execute(
            "INSERT OR IGNORE INTO votes (job_id, wallet, signature, created_at) VALUES (?, ?, NULL, ?)",
            (job_id, wallet, time.time()),
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
