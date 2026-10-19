"""One USDC tx pays for one thing suite-wide (memestudio's ledger), /parse is capped per wallet,
and request fields have sane lengths."""
import json
import sqlite3
import time
from datetime import datetime, timezone

import api
from conftest import INTERNAL_SECRET
from test_identity import RAW, as_, client, credit_token, new_wallet


# ── suite tx ledger ──────────────────────────────────────────────────────────

def test_ledger_claim_accepts_once_then_409_is_already_used(stubs):
    me = new_wallet()
    first = client.post("/generate/raw", json={**RAW, "tx_signature": "tx-ledger-1"}, headers=as_(me))
    assert first.status_code == 202, first.text
    assert stubs["claims"] == [{"tx_signature": "tx-ledger-1", "wallet": me, "purpose": "factory_generate"}]

    again = client.post("/generate/raw", json={**RAW, "tx_signature": "tx-ledger-1"}, headers=as_(me))
    assert again.status_code == 402 and again.json()["detail"] == "Transaction already used"
    assert len(stubs["jobs"]) == 1


def test_tx_claimed_elsewhere_buys_nothing_here(stubs, monkeypatch):
    # e.g. the same tx already bought Studio credits
    monkeypatch.setattr(api, "_post_tx_claim", lambda body: 409)
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx-studio"}, headers=as_(new_wallet()))
    assert r.status_code == 402 and r.json()["detail"] == "Transaction already used"
    assert stubs["jobs"] == []


def test_ledger_unreachable_fails_closed_and_frees_the_tx(stubs, monkeypatch):
    def consume(tx_signature, wallet):
        con = sqlite3.connect(api.DB_PATH)
        con.execute("INSERT INTO used_transactions VALUES (?, ?, ?)", (tx_signature, wallet, time.time()))
        con.commit()
        con.close()
        return True, ""

    def down(body):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(api, "verify_usdc_payment", consume)
    monkeypatch.setattr(api, "_post_tx_claim", down)
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx-down"}, headers=as_(new_wallet()))
    assert r.status_code == 503
    assert stubs["jobs"] == []
    # The member can retry the same tx once memestudio is back
    con = sqlite3.connect(api.DB_PATH)
    assert con.execute("SELECT 1 FROM used_transactions WHERE tx_signature = 'tx-down'").fetchone() is None
    con.close()


def test_ledger_server_error_fails_closed(stubs, monkeypatch):
    monkeypatch.setattr(api, "_post_tx_claim", lambda body: 500)
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx-500"}, headers=as_(new_wallet()))
    assert r.status_code == 503 and stubs["jobs"] == []


def test_no_internal_secret_means_no_usdc_but_credits_still_work(stubs, monkeypatch):
    monkeypatch.setattr(api, "_VVC_INTERNAL_SECRET", None)
    me = new_wallet()
    r = client.post("/generate/raw", json={**RAW, "tx_signature": "tx-nosecret"}, headers=as_(me))
    assert r.status_code == 503
    assert stubs["usdc"] == []  # refused before touching Solana
    ok = client.post("/generate/raw", json={**RAW, "credit_token": credit_token(me)}, headers=as_(me))
    assert ok.status_code == 202, ok.text
    assert stubs["claims"] == []


def test_claim_request_shape(monkeypatch):
    seen = {}

    class Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout):
        seen.update(url=req.full_url, secret=req.get_header("X-vvc-internal"), body=json.loads(req.data))
        return Resp()

    monkeypatch.setattr(api.urllib.request, "urlopen", fake_urlopen)
    assert api._post_tx_claim({"tx_signature": "t", "wallet": "w", "purpose": "factory_generate"}) == 200
    assert seen["url"].endswith("/internal/tx/claim")
    assert seen["secret"] == INTERNAL_SECRET
    assert seen["body"]["purpose"] == "factory_generate"


# ── /parse limits ────────────────────────────────────────────────────────────

def test_parse_is_capped_per_wallet_per_10_minutes(stubs):
    me = new_wallet()
    for _ in range(20):
        assert client.post("/parse", json={"text": "cat vs dog"}, headers=as_(me)).status_code == 200
    r = client.post("/parse", json={"text": "cat vs dog"}, headers=as_(me))
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert len(stubs["parsed"]) == 20  # the 21st never reached Gemini
    # Someone else is unaffected
    assert client.post("/parse", json={"text": "cat vs dog"}, headers=as_(new_wallet())).status_code == 200
    # Once the window has passed, the wallet can parse again
    api._parse_recent[me] = [t - 601 for t in api._parse_recent[me]]
    assert client.post("/parse", json={"text": "cat vs dog"}, headers=as_(me)).status_code == 200


def test_parse_is_capped_per_wallet_per_day(stubs):
    me = new_wallet()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    con = sqlite3.connect(api.DB_PATH)
    con.execute("INSERT INTO parse_usage (wallet, day, count) VALUES (?, ?, 199)", (me, day))
    con.commit()
    con.close()
    assert client.post("/parse", json={"text": "cat vs dog"}, headers=as_(me)).status_code == 200
    r = client.post("/parse", json={"text": "cat vs dog"}, headers=as_(me))
    assert r.status_code == 429 and "Daily" in r.json()["detail"]


# ── field lengths ────────────────────────────────────────────────────────────

def test_archetypes_and_labels_have_length_caps(stubs):
    me = new_wallet()
    pay = {"credit_token": credit_token(me)}
    assert client.post("/generate/raw", json={"virgin": "x" * 101, "chad": "Dog", **pay}, headers=as_(me)).status_code == 422
    assert client.post("/generate/raw", json={**RAW, "chad_labels": ["y" * 121], **pay}, headers=as_(me)).status_code == 422
    assert client.post("/generate", json={**RAW, "virgin_labels": ["a"], "chad_labels": ["b" * 121], **pay}, headers=as_(me)).status_code == 422
    assert stubs["jobs"] == []
    ok = client.post("/generate/raw", json={"virgin": "x" * 100, "chad": "Dog", "chad_labels": ["y" * 120], **pay}, headers=as_(me))
    assert ok.status_code == 202, ok.text
