"""Meme Factory acts as the identity session's wallet: reads public, generating/parsing/voting for
members, and every generation paid by that member (USDC tx or a single-use credit token)."""
import hashlib
import hmac
import sqlite3
import time
import uuid

import pytest
from fastapi.testclient import TestClient

import api
from conftest import CREDIT_SECRET, ORIGIN, make_token

client = TestClient(api.app)


def new_wallet() -> str:
    return f"W{uuid.uuid4().hex}"


def as_(wallet=None, member=True, origin=ORIGIN, ttl=900):
    headers = {"origin": origin}
    if wallet:
        headers["cookie"] = f"vvc_at={make_token(wallet, member, ttl)}"
    return headers


def credit_token(wallet, ts=None, secret=CREDIT_SECRET):
    ts = ts if ts is not None else int(time.time() * 1000)
    sig = hmac.new(secret.encode(), f"{wallet}|{ts}|meme_generate".encode(), hashlib.sha256).hexdigest()
    return {"wallet": wallet, "ts": ts, "sig": sig}


RAW = {"virgin": "Cat", "chad": "Dog"}


# ── who may do what ──────────────────────────────────────────────────────────

def test_reads_are_public():
    assert client.get("/health").status_code == 200
    assert client.get("/memes").status_code == 200


def test_guests_and_pending_cannot_generate(stubs):
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx"}, headers=as_())
    assert r.status_code == 401 and r.json()["code"] == "signin_required"
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx"}, headers=as_(new_wallet(), member=False))
    assert r.status_code == 403 and r.json()["code"] == "membership_required"
    assert stubs["usdc"] == [] and stubs["jobs"] == []


def test_expired_session_says_so(stubs):
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx"}, headers=as_(new_wallet(), ttl=-3600))
    assert r.status_code == 401 and r.json()["code"] == "token_expired"


def test_cross_site_cookie_writes_are_refused(stubs):
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx"}, headers=as_(new_wallet(), origin="https://evil.example"))
    assert r.status_code == 403 and r.json()["code"] == "bad_origin"


def test_parse_costs_an_llm_call_so_members_only(stubs):
    assert client.post("/parse", json={"text": "cat vs dog"}, headers=as_()).status_code == 401
    assert client.post("/parse", json={"text": "cat vs dog"}, headers=as_(new_wallet())).status_code == 200


# ── payments are the member's ────────────────────────────────────────────────

def test_usdc_payment_is_checked_against_the_session_wallet(stubs):
    me = new_wallet()
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx-1", "wallet": "someone-else"}, headers=as_(me))
    assert r.status_code == 202, r.text
    assert stubs["usdc"] == [("tx-1", me)]
    assert stubs["jobs"] == [{"wallet": me, "payment": "tx-1"}]


def test_no_payment_no_meme(stubs):
    r = client.post("/generate/raw", json=RAW, headers=as_(new_wallet()))
    assert r.status_code == 402
    assert stubs["jobs"] == []


def test_credit_token_works_once(stubs):
    me = new_wallet()
    tok = credit_token(me)
    assert client.post("/generate/raw", json={**RAW, "credit_token": tok}, headers=as_(me)).status_code == 202
    again = client.post("/generate/raw", json={**RAW, "credit_token": tok}, headers=as_(me))
    assert again.status_code == 402 and "already used" in again.json()["detail"]
    assert len(stubs["jobs"]) == 1 and stubs["jobs"][0]["payment"].startswith("credit:")


@pytest.mark.parametrize("make,why", [
    (lambda me: credit_token(new_wallet()), "another wallet"),
    (lambda me: credit_token(me, ts=int(time.time() * 1000) - 10 * 60 * 1000), "expired"),
    (lambda me: credit_token(me, secret="forged"), "Invalid"),
    (lambda me: {**credit_token(me), "ts": int(time.time() * 1000) + 5}, "Invalid"),
])
def test_bad_credit_tokens_are_refused(stubs, make, why):
    me = new_wallet()
    r = client.post("/generate/raw", json={**RAW, "credit_token": make(me)}, headers=as_(me))
    assert r.status_code == 402
    assert why in r.json()["detail"]
    assert stubs["jobs"] == []


def test_freestyle_parses_before_taking_payment(stubs):
    me = new_wallet()
    tok = credit_token(me)
    bad = client.post("/generate/freestyle", json={"text": "nonsense", "credit_token": tok}, headers=as_(me))
    assert bad.status_code == 422
    # The token wasn't spent on text we couldn't use
    ok = client.post("/generate/freestyle", json={"text": "cat vs dog", "credit_token": tok}, headers=as_(me))
    assert ok.status_code == 202, ok.text


# ── votes ────────────────────────────────────────────────────────────────────

def _insert_meme(job_id):
    con = sqlite3.connect(api.DB_PATH)
    con.execute("INSERT INTO memes (job_id, status, created_at) VALUES (?, 'done', ?)", (job_id, time.time()))
    con.commit()
    con.close()


def test_votes_are_cast_as_the_member_once():
    job = f"api_vote_{uuid.uuid4().hex[:8]}"
    _insert_meme(job)
    me = new_wallet()
    assert client.post(f"/jobs/{job}/vote", json={"wallet": "someone-else"}, headers=as_()).status_code == 401
    first = client.post(f"/jobs/{job}/vote", json={"wallet": "someone-else", "signature": "x"}, headers=as_(me))
    assert first.status_code == 200 and first.json() == {"job_id": job, "vote_count": 1, "already_voted": False}
    second = client.post(f"/jobs/{job}/vote", headers=as_(me))
    assert second.json()["already_voted"] is True
    con = sqlite3.connect(api.DB_PATH)
    wallets = [r[0] for r in con.execute("SELECT wallet FROM votes WHERE job_id = ?", (job,))]
    con.close()
    assert wallets == [me]
