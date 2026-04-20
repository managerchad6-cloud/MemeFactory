import os
import json
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# ======================
# ENV
# ======================

load_dotenv()

# ======================
# CONFIG
# ======================

MODEL = "gpt-5.2"
CONTEXT_FILE = "context.json"

# Support environment variable override for job isolation
IDEAS_FILE = os.environ.get("IDEAS_FILE", "ideas.json")

MIN_LABELS = 5
MAX_LABELS = 8

SYSTEM_PROMPT = """You generate Virgin vs Chad meme prompts.

You are NOT allowed to output JSON.
You are NOT allowed to explain anything.
You ONLY output the final Nanobanana prompt text.

Rules:
- Respect ALL constraints in the provided context.
- Focus on concrete, drawable visual traits.
- Never use vague phrases like "simple", "basic", "unchanged", or "context-appropriate".
- Always include flat color fills and balanced negative space.
- Be specific, ugly, and meme-correct.
"""

# ======================
# HELPERS
# ======================

def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("__", "_")
    )


def normalize_archetype(text: str) -> str:
    t = text.strip().lower()
    for prefix in ("virgin", "chad"):
        if t.startswith(prefix + " "):
            t = t[len(prefix) + 1 :]
    return t.strip().title()


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def save_single_idea(path, idea: dict):
    """Save idea to path (supports both str and Path)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"items": [idea]}, indent=2),
        encoding="utf-8"
    )


def clamp_labels(labels: List[str]) -> List[str]:
    return labels[:MAX_LABELS]


# ======================
# AI TEXT GENERATION
# ======================

def generate_reskin_prompt(
    client: OpenAI,
    context: str,
    virgin: str,
    chad: str,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
        {
            "role": "user",
            "content": f"""
Virgin archetype: {virgin}
Chad archetype: {chad}

Write the FULL Nanobanana IMAGE RESKIN PROMPT for the FIRST image pass.

REQUIREMENTS:
- Use the standard Virgin vs Chad base template
- Absolute pose, face, and composition locks
- Facial proportions and expressions must remain IDENTICAL
- Concrete, drawable clothing, props, accessories, and cosmetic modifications ONLY
- Mandatory flat color fills (no black-and-white, no grayscale)
- Intentionally ugly, flat, sketchy, meme-like style
- Balanced negative space on BOTH sides for later text placement
- Avoid crowding the center gap between characters

TEXT REQUIREMENTS:
- Render title text above each character
- EXACT text:
  - "Virgin {virgin}"
  - "Chad {chad}"
- Title text must not overlap heads or faces
- Plain, simple lettering only (no polish)

Do NOT add any additional text.
Do NOT alter the background.
This is the FIRST image pass only.

Output ONLY the prompt text.
"""
        },
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return resp.choices[0].message.content.strip()

def resolve_labels(
    client: OpenAI,
    context: str,
    virgin: str,
    chad: str,
    virgin_labels: List[str],
    chad_labels: List[str],
) -> tuple:
    """
    Return the full label lists (5–8 per character).
    If user already provided >= MIN_LABELS, return them clamped.
    Otherwise call the LLM to fill in the gaps and return the complete set.
    """
    v_full = len(virgin_labels) >= MIN_LABELS
    c_full = len(chad_labels) >= MIN_LABELS

    if v_full and c_full:
        return clamp_labels(virgin_labels), clamp_labels(chad_labels)

    def label_section(side, labels):
        if len(labels) >= MIN_LABELS:
            return (
                f"{side} — already complete, use these exactly:\n"
                + "\n".join(f"- {l}" for l in labels[:MAX_LABELS])
            )
        if labels:
            return (
                f"{side} — keep these verbatim, add {MIN_LABELS - len(labels)}–{MAX_LABELS - len(labels)} more:\n"
                + "\n".join(f"- {l}" for l in labels)
            )
        return f"{side} — generate {MIN_LABELS}–{MAX_LABELS} labels."

    prompt = f"""
Virgin archetype: {virgin}
Chad archetype: {chad}

{label_section("VIRGIN", virgin_labels)}

{label_section("CHAD", chad_labels)}

Return a JSON object with exactly two keys: "virgin_labels" and "chad_labels".
Each value is an array of strings (the complete final label list).
Output ONLY valid JSON. No commentary, no markdown.
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You generate Virgin vs Chad meme labels. Output ONLY valid JSON."},
            {"role": "user", "content": context},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(resp.choices[0].message.content)
    full_virgin = clamp_labels(data.get("virgin_labels", virgin_labels))
    full_chad = clamp_labels(data.get("chad_labels", chad_labels))
    return full_virgin, full_chad


def generate_annotation_prompt(
    client: OpenAI,
    context: str,
    virgin: str,
    chad: str,
    virgin_labels: List[str],
    chad_labels: List[str],
) -> str:
    def label_instruction(side: str, labels: List[str]) -> str:
        if not labels:
            return (
                f"- Generate {MIN_LABELS}–{MAX_LABELS} labels.\n"
                f"- Prefer fewer labels unless more are clearly justified."
            )


        if len(labels) >= MIN_LABELS:
            return (
                f"- Use these labels verbatim:\n"
                + "\n".join(f"  - {l}" for l in labels)
            )
        return (
            f"- These labels are fixed and verbatim:\n"
            + "\n".join(f"  - {l}" for l in labels)
            + f"\n- Generate {MIN_LABELS - len(labels)}–{MAX_LABELS - len(labels)} additional labels."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
        {
            "role": "user",
            "content": f"""
Virgin archetype: {virgin}
Chad archetype: {chad}

Write the FULL Nanobanana TEXT-ANNOTATION PROMPT for the SECOND image pass.

ABSOLUTE RULES:
- Do NOT modify the image content
- Preserve title text exactly
- Arial font only, black text

LAYOUT:
- Diagram-style
- Labels distributed around head / torso / legs
- NO single vertical column
- Left stays left, right stays right

VIRGIN LABELS:
{label_instruction("virgin", virgin_labels)}

CHAD LABELS:
{label_instruction("chad", chad_labels)}

Output ONLY the annotation prompt text.
"""
        },
    ]

    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content.strip()

def read_input_payload() -> dict:
    raw = sys.stdin.read().strip()

    if not raw:
        raise RuntimeError("No input provided")

    # JSON payload (from Telegram bot)
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError("Invalid JSON input")

    # Legacy CLI input: "Virgin X, Chad Y"
    if "," not in raw:
        raise RuntimeError("Format must be: Virgin X, Chad Y")

    left, right = [p.strip() for p in raw.split(",", 1)]
    virgin = normalize_archetype(left.replace("Virgin", ""))
    chad = normalize_archetype(right.replace("Chad", ""))

    return {
        "virgin": virgin,
        "chad": chad,
        "virgin_labels": [],
        "chad_labels": [],
    }


# ======================
# CORE FUNCTION
# ======================

def generate_idea(
    virgin: str,
    chad: str,
    virgin_labels: Optional[List[str]] = None,
    chad_labels: Optional[List[str]] = None,
):
    virgin_labels = clamp_labels(virgin_labels or [])
    chad_labels = clamp_labels(chad_labels or [])

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    base_dir = Path(__file__).parent
    context = load_text(base_dir / CONTEXT_FILE)

    print("Generating reskin prompt…")
    reskin_prompt = generate_reskin_prompt(
        client, context, virgin, chad
    )

    print("Resolving full label lists…")
    full_virgin_labels, full_chad_labels = resolve_labels(
        client, context, virgin, chad, virgin_labels, chad_labels
    )

    print("Generating annotation prompt…")
    annotation_prompt = generate_annotation_prompt(
        client, context, virgin, chad, full_virgin_labels, full_chad_labels
    )

    idea = {
        "id": slugify(f"virgin_{virgin}_vs_chad_{chad}"),
        "reskin_prompt": reskin_prompt,
        "annotation_prompt": annotation_prompt,
        "virgin_labels": full_virgin_labels,
        "chad_labels": full_chad_labels,
    }

    # Use IDEAS_FILE which may be overridden by environment variable
    ideas_file_path = Path(IDEAS_FILE) if not Path(IDEAS_FILE).is_absolute() else Path(IDEAS_FILE)
    if not ideas_file_path.is_absolute():
        ideas_file_path = base_dir / ideas_file_path

    save_single_idea(ideas_file_path, idea)
    return idea


# ======================
# CLI ENTRYPOINT
# ======================

def main():
    print("Virgin vs Chad prompt generator")
    print("--------------------------------")

    payload = read_input_payload()

    virgin = normalize_archetype(payload["virgin"])
    chad = normalize_archetype(payload["chad"])

    virgin_labels = payload.get("virgin_labels", [])
    chad_labels = payload.get("chad_labels", [])

    idea = generate_idea(
        virgin=virgin,
        chad=chad,
        virgin_labels=virgin_labels,
        chad_labels=chad_labels,
    )

    print("\nSUCCESS")
    print(f"Generated idea: {idea['id']}")

if __name__ == "__main__":
    main()
