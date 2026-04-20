import os
import sys
import subprocess
import json
import time
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ======================
# ENV / PATHS
# ======================

load_dotenv()

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

BASE_DIR = Path(__file__).parent
JOBS_DIR = BASE_DIR / "jobs"

GENERATE_IDEA_SCRIPT = BASE_DIR / "generate_idea.py"
RUN_BATCH_SCRIPT = BASE_DIR / "run_batch.py"

# Ensure jobs directory exists
JOBS_DIR.mkdir(exist_ok=True)

# ======================
# USER SESSION STATE
# ======================

USER_STATE = {}
# USER_STATE[user_id] = {
#   "stage": "awaiting_virgin_labels" | "awaiting_chad_labels",
#   "virgin": str,
#   "chad": str,
#   "virgin_labels": [],
#   "chad_labels": []
# }

MAX_LABELS = 12

# ======================
# HELPERS
# ======================

def create_job_id(user_id: int) -> str:
    """Create unique job ID: <user_id>_<timestamp>"""
    timestamp = int(time.time())
    return f"{user_id}_{timestamp}"


def create_job_workspace(job_id: str) -> tuple[Path, Path, Path]:
    """
    Create isolated workspace for a job.
    Returns: (job_dir, ideas_file, out_dir)
    """
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ideas_file = job_dir / "ideas.json"
    out_dir = job_dir / "out"
    out_dir.mkdir(exist_ok=True)

    return job_dir, ideas_file, out_dir


def get_job_output_image(out_dir: Path) -> Path | None:
    """Get the most recent image from a job's output directory"""
    if not out_dir.exists():
        return None
    images = list(out_dir.glob("*.png"))
    if not images:
        return None
    return max(images, key=lambda p: p.stat().st_mtime)


def reset_user(user_id: int):
    USER_STATE.pop(user_id, None)


async def run_meme_generation(
    update: Update,
    payload: dict,
    ideas_file: Path,
    out_dir: Path,
):
    """
    Asynchronous background task to generate meme without blocking Telegram handlers.
    Runs subprocesses with environment variable overrides for job isolation.
    """
    user_id = update.effective_user.id

    # Prepare environment with job-specific paths
    env = os.environ.copy()
    env["IDEAS_FILE"] = str(ideas_file.absolute())
    env["OUT_DIR"] = str(out_dir.absolute())

    try:
        # Run generate_idea.py
        process1 = await asyncio.create_subprocess_exec(
            sys.executable,
            str(GENERATE_IDEA_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout1, stderr1 = await process1.communicate(input=json.dumps(payload).encode())

        if process1.returncode != 0:
            error_msg = stderr1.decode() if stderr1 else "Unknown error"
            await update.message.reply_text(f"Error generating idea: {error_msg[:200]}")
            reset_user(user_id)
            return

        # Run run_batch.py
        process2 = await asyncio.create_subprocess_exec(
            sys.executable,
            str(RUN_BATCH_SCRIPT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout2, stderr2 = await process2.communicate()

        if process2.returncode != 0:
            error_msg = stderr2.decode() if stderr2 else "Unknown error"
            await update.message.reply_text(f"Error generating image: {error_msg[:200]}")
            reset_user(user_id)
            return

        # Success - send the image
        image_path = get_job_output_image(out_dir)
        if image_path:
            with open(image_path, "rb") as f:
                await update.message.reply_photo(photo=f)
        else:
            await update.message.reply_text("No image produced.")

    except Exception as e:
        await update.message.reply_text(f"Unexpected error: {str(e)[:200]}")

    finally:
        reset_user(user_id)


# ======================
# BOT HANDLERS
# ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send your meme idea in this format:\n\nVirgin X, Chad Y"
    )


async def next_step(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in USER_STATE:
        await update.message.reply_text("Nothing to advance.")
        return

    state = USER_STATE[user_id]

    # VIRGIN → CHAD
    if state["stage"] == "awaiting_virgin_labels":
        state["stage"] = "awaiting_chad_labels"
        await update.message.reply_text(
            "Enter Chad labels (up to 12).\n"
            "Send one label per message.\n"
            "Send /next when finished."
        )
        return

    # CHAD → GENERATE
    if state["stage"] == "awaiting_chad_labels":
        await update.message.reply_text("Generating meme…")

        # Create unique job workspace
        job_id = create_job_id(user_id)
        job_dir, ideas_file, out_dir = create_job_workspace(job_id)

        payload = {
            "virgin": state["virgin"],
            "chad": state["chad"],
            "virgin_labels": state["virgin_labels"],
            "chad_labels": state["chad_labels"],
        }

        # Launch async background task (non-blocking)
        asyncio.create_task(
            run_meme_generation(update, payload, ideas_file, out_dir)
        )
        return


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text.strip()

    # ------------------
    # NEW SESSION
    # ------------------
    if user_id not in USER_STATE:
        if "," not in text or not text.lower().startswith("virgin"):
            await update.message.reply_text(
                "Invalid format.\n\nUse:\nVirgin X, Chad Y"
            )
            return

        left, right = [p.strip() for p in text.split(",", 1)]
        virgin = left.replace("Virgin", "").strip()
        chad = right.replace("Chad", "").strip()

        USER_STATE[user_id] = {
            "stage": "awaiting_virgin_labels",
            "virgin": virgin,
            "chad": chad,
            "virgin_labels": [],
            "chad_labels": [],
        }

        await update.message.reply_text(
            "Enter Virgin labels (up to 12).\n"
            "Send one label per message.\n"
            "Send /next when finished."
        )
        return

    state = USER_STATE[user_id]

    # ------------------
    # VIRGIN LABELS
    # ------------------
    if state["stage"] == "awaiting_virgin_labels":
        if text.startswith("/"):
            await update.message.reply_text("Use /next when finished.")
            return

        if len(state["virgin_labels"]) >= MAX_LABELS:
            await update.message.reply_text("Maximum 8 labels reached.")
            return

        state["virgin_labels"].append(text)
        return

    # ------------------
    # CHAD LABELS
    # ------------------
    if state["stage"] == "awaiting_chad_labels":
        if text.startswith("/"):
            await update.message.reply_text("Use /next when finished.")
            return

        if len(state["chad_labels"]) >= MAX_LABELS:
            await update.message.reply_text("Maximum 8 labels reached.")
            return

        state["chad_labels"].append(text)
        return


# ======================
# MAIN
# ======================

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("next", next_step))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram bot running…")
    app.run_polling()


if __name__ == "__main__":
    main()
