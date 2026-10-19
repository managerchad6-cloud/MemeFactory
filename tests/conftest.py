"""Boots api.py against a temp DB with a test identity key; network-bound parts are stubbed."""
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ORIGIN = "http://localhost:5291"
CREDIT_SECRET = "test-credit-secret"
INTERNAL_SECRET = "test-internal-secret"

_tmp = Path(tempfile.mkdtemp(prefix="mf-test-"))
_key = Ed25519PrivateKey.generate()
(_tmp / "identity.pub.pem").write_bytes(_key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))

os.environ.update({
    "MF_DB_PATH": str(_tmp / "memes.db"),
    "IDENTITY_PUBLIC_KEY_PATH": str(_tmp / "identity.pub.pem"),
    "IDENTITY_ALLOWED_ORIGINS": ORIGIN,
    "CREDIT_SHARED_SECRET": CREDIT_SECRET,
    "VVC_INTERNAL_SECRET": INTERNAL_SECRET,
    "OPENAI_API_KEY": "test",
})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api  # noqa: E402


def _b64(o) -> str:
    return base64.urlsafe_b64encode(json.dumps(o).encode()).rstrip(b"=").decode()


def make_token(wallet: str, member: bool = True, ttl: int = 900) -> str:
    """Access token as the identity service mints it."""
    now = int(time.time())
    head = _b64({"alg": "EdDSA", "typ": "JWT"})
    body = _b64({"iss": "vvc-identity", "aud": "vvc", "sub": wallet, "sid": "t", "mem": member, "iat": now, "exp": now + ttl})
    sig = base64.urlsafe_b64encode(_key.sign(f"{head}.{body}".encode())).rstrip(b"=").decode()
    return f"{head}.{body}.{sig}"


@pytest.fixture
def stubs(monkeypatch):
    """Record payments/jobs instead of touching Solana, memestudio, OpenAI or the image pipeline."""
    calls = {"usdc": [], "jobs": [], "parsed": [], "claims": []}

    def fake_usdc(tx_signature, wallet):
        calls["usdc"].append((tx_signature, wallet))
        return (tx_signature != "bad-tx", "" if tx_signature != "bad-tx" else "Transaction was not signed by the claimed wallet")

    def fake_submit(virgin, chad, vl, cl, wallet=None, tx_signature=None):
        calls["jobs"].append({"wallet": wallet, "payment": tx_signature})
        return f"api_test_{len(calls['jobs'])}"

    def fake_parse(text):
        if text == "nonsense":
            raise api.HTTPException(status_code=422, detail="Could not extract both archetypes from text")
        calls["parsed"].append(text)
        return {"virgin": "A", "chad": "B", "virgin_labels": [], "chad_labels": []}

    def fake_claim(body):
        # The suite ledger: first claim of a tx wins, any later one is a 409
        seen = [c["tx_signature"] for c in calls["claims"]]
        calls["claims"].append(body)
        return 409 if body["tx_signature"] in seen else 200

    monkeypatch.setattr(api, "verify_usdc_payment", fake_usdc)
    monkeypatch.setattr(api, "_post_tx_claim", fake_claim)
    monkeypatch.setattr(api, "_submit_job", fake_submit)
    monkeypatch.setattr(api, "_parse_freestyle", fake_parse)
    return calls
