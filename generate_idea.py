import os
import json
import sys
import random
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

MIN_LABELS = 4
MAX_LABELS = 7

# ======================
# STYLE RANDOMISATION
# ======================

_ART_STYLES = [
    "extremely rough digital pencil sketch feel — uneven wobbly lines, scratchy cross-hatching in shadows, nothing is straight, all on a plain white canvas",
    "MS Paint tier — thick jagged outlines, solid block fills, zero anti-aliasing, colours slightly off-model like someone used the paint bucket wrong",
    "crude digital marker drawing — thick felt-tip style outlines with visible stroke direction, some areas filled with hasty overlapping strokes, not clean at all",
    "mid-2000s DeviantArt fan-art energy — slightly cleaner than pure scribble but still flat and obviously meme-tier, soft outlines, white background",
    "quick digital doodle feel — thin scratchy lines, minimal fill, some areas just left white, very impatient energy, like drawn fast in a free app",
    "Microsoft Word clip-art meets meme — overly smooth outlines but comically off-proportion, fills too saturated, everything slightly wrong",
]

_TITLE_STYLES = [
    "bold all-caps Impact-style font, titles slightly different sizes on each side as if typed separately",
    "chunky hand-lettered block capitals, slightly uneven baseline, letters not perfectly spaced",
    "comic-book style bold lettering, slightly slanted, thick black outline around the text",
    "plain sans-serif but sized inconsistently — one side's title noticeably bigger than the other",
    "scratchy hand-written capitals, not horizontal, one word slightly higher than the next",
    "big bold text but clearly done in two different 'handwriting' styles — like two people labelled their own side",
]

_LABEL_TEXT_STYLES = [
    "all labels in small plain text, connected by thin straight lines with a tiny arrowhead",
    "labels in slightly varying font sizes — some bigger for emphasis, no consistent sizing, feels unplanned",
    "labels in a slightly cramped handwritten style, lines are freehand and wobbly, not ruler-straight",
    "mix of short and long pointer lines, some labels far from the character, some nearly touching it",
    "no pointer lines at all — labels just float near the relevant body part with implied proximity",
    "labels in a slightly bold style, pointer lines with a small dot at the character end instead of an arrow",
]

_LABEL_LAYOUTS = [
    "most labels clustered around the upper body and head, only one or two near the feet",
    "evenly spread head-to-toe on both sides, labels at roughly equal vertical intervals",
    "more labels near the head and feet, fewer around the torso — sparse in the middle",
    "asymmetric — virgin side labels mostly on the left outer edge, chad side labels closer in and scattered",
    "labels spread wide outward, the outermost ones almost at the image edge, creating a diagram-like exploded view",
    "dense near the torso, with labels slightly overlapping each other's pointer lines — crowded, organic feel",
]

_COLOR_APPROACHES = [
    "flat solid colours, fairly saturated and slightly garish — too many colours, no restraint",
    "limited palette: only 3–4 colours per character, feels like someone ran out of digital swatches",
    "muted desaturated colours, slightly washed out, like someone turned the saturation slider down too far",
    "bright oversaturated fills, almost neon in places — definitely chosen by someone who loved the colour wheel",
    "mostly flat but with a single rough shadow colour per character, added as a solid dark shape, no blending",
]

# Phrasing modes with target weights:
#   muy_cortitos  20% — 2-3 word max fragments
#   current       40% — 4-7 words, like existing outputs
#   some_long     10% — mostly current length but 2-3 labels per side are full long sentences
#   medium_plus   20% — 6-10 words, a step up from current
#   full_mix      10% — genuine mix of very short, medium, and long in the same meme

_LABEL_PHRASING_MODES = [
    ("muy_cortitos", 20),
    ("current",      40),
    ("some_long",    10),
    ("medium_plus",  20),
    ("full_mix",     10),
]

_LABEL_PHRASING_INSTRUCTIONS = {
    "muy_cortitos": (
        "extremely short fragments only — 2 to 3 words maximum per label, no exceptions. "
        "Single noun phrases or verb fragments. "
        "Examples: 'skill issue', 'no car', 'still at mom\\'s', 'cries at ads', 'always late'."
    ),
    "current": (
        "short punchy phrases — 4 to 7 words each. "
        "Examples: 'needs caffeine to function', 'circles for parking like a vulture', "
        "'owns three chisels he never uses'."
    ),
    "medium_plus": (
        "medium-length descriptive phrases — 6 to 10 words each, slightly more specific than average. "
        "Examples: 'still checking the weather app before leaving anyway', "
        "'bought a standing desk, sits on a stool'."
    ),
    "some_long": (
        "mostly short punchy phrases (4–7 words), BUT exactly 2 or 3 labels per character must be "
        "full long sentences of 10 to 15 words. The long ones should be the most specific and funny. "
        "Mix freely — the long ones can appear anywhere in the list."
    ),
    "full_mix": (
        "a genuine mix of all lengths in the same meme: "
        "at least 1 label per character must be very short (2–3 words), "
        "at least 1 must be medium (6–10 words), "
        "and at least 1 must be a long full sentence (10–15 words). "
        "The rest can be any length. Do not group them by length."
    ),
}


def pick_style() -> dict:
    """Randomly select one option per style dimension for this generation."""
    modes, weights = zip(*_LABEL_PHRASING_MODES)
    phrasing = random.choices(modes, weights=weights, k=1)[0]
    label_count = random.randint(MIN_LABELS, MAX_LABELS)
    return {
        "art_style":          random.choice(_ART_STYLES),
        "title_style":        random.choice(_TITLE_STYLES),
        "label_text_style":   random.choice(_LABEL_TEXT_STYLES),
        "label_layout":       random.choice(_LABEL_LAYOUTS),
        "color_approach":     random.choice(_COLOR_APPROACHES),
        "label_phrasing":     phrasing,
        "label_phrasing_instruction": _LABEL_PHRASING_INSTRUCTIONS[phrasing],
        "label_count":        label_count,
    }


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
    style: dict,
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
- Balanced negative space on BOTH sides for later text placement
- Avoid crowding the center gap between characters
- DIGITAL DRAWING ONLY — plain white background, no paper texture, no notebook lines, no grain, no physical media simulation

STYLE FOR THIS GENERATION (follow these specifically — they define the organic feel):
- Art style: {style["art_style"]}
- Colour approach: {style["color_approach"]}

TEXT REQUIREMENTS:
- Render title text above each character
- EXACT text:
  - "Virgin {virgin}"
  - "Chad {chad}"
- Title lettering style: {style["title_style"]}
- Title text must not overlap heads or faces

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
    style: dict,
) -> tuple:
    """
    Return the full label lists for this generation.
    Count and phrasing style come from the style dict so each run varies.
    If user already provided enough labels, return them clamped (phrasing untouched).
    """
    target = style["label_count"]
    phrasing_instruction = style["label_phrasing_instruction"]

    v_full = len(virgin_labels) >= target
    c_full = len(chad_labels) >= target

    if v_full and c_full:
        return virgin_labels[:target], chad_labels[:target]

    def label_section(side, labels):
        if len(labels) >= target:
            return (
                f"{side} — already complete, use these exactly:\n"
                + "\n".join(f"- {l}" for l in labels[:target])
            )
        if labels:
            return (
                f"{side} — keep these verbatim, add {target - len(labels)} more:\n"
                + "\n".join(f"- {l}" for l in labels)
            )
        return f"{side} — generate exactly {target} labels."

    prompt = f"""
Virgin archetype: {virgin}
Chad archetype: {chad}

LABEL PHRASING STYLE (apply to all generated labels): {phrasing_instruction}

{label_section("VIRGIN", virgin_labels)}

{label_section("CHAD", chad_labels)}

Return a JSON object with exactly two keys: "virgin_labels" and "chad_labels".
Each value must be an array of exactly {target} strings.
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
    full_virgin = data.get("virgin_labels", virgin_labels)[:target]
    full_chad = data.get("chad_labels", chad_labels)[:target]
    return full_virgin, full_chad


def generate_annotation_prompt(
    client: OpenAI,
    context: str,
    virgin: str,
    chad: str,
    virgin_labels: List[str],
    chad_labels: List[str],
    style: dict,
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
- Black text

STYLE FOR THIS GENERATION (carry these through from the first pass):
- Label text style: {style["label_text_style"]}
- Label spatial layout: {style["label_layout"]}

LAYOUT:
- Left stays left, right stays right
- Do NOT stack all labels into a single vertical column

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

    style = pick_style()
    print(f"Style: art={style['art_style'][:40]}… | title={style['title_style'][:40]}…")

    print("Generating reskin prompt…")
    reskin_prompt = generate_reskin_prompt(
        client, context, virgin, chad, style
    )

    print("Resolving full label lists…")
    full_virgin_labels, full_chad_labels = resolve_labels(
        client, context, virgin, chad, virgin_labels, chad_labels, style
    )

    print("Generating annotation prompt…")
    annotation_prompt = generate_annotation_prompt(
        client, context, virgin, chad, full_virgin_labels, full_chad_labels, style
    )

    idea = {
        "id": slugify(f"virgin_{virgin}_vs_chad_{chad}"),
        "reskin_prompt": reskin_prompt,
        "annotation_prompt": annotation_prompt,
        "virgin_labels": full_virgin_labels,
        "chad_labels": full_chad_labels,
        "style": style,
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
