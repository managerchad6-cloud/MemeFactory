import os
import json
import re
import time
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from google import genai
from google.genai import types


# ======================
# ENV
# ======================

load_dotenv()

# ======================
# CONFIG
# ======================

MODEL = "gemini-3.1-flash-image-preview"

BASE_IMAGE = "base.png"

# Support environment variable overrides for job isolation
IDEAS_FILE = os.environ.get("IDEAS_FILE", "ideas.json")
OUT_DIR = os.environ.get("OUT_DIR", "out")

PASS2_RETRIES = 3
PASS2_DELAY_SECONDS = 2


# ======================
# LOGGING SETUP
# ======================

def setup_logging():
    """Configure logging to both file and console with detailed formatting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"run_batch_{timestamp}.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # File handler - logs everything
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - logs INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging initialized - log file: {log_file}")
    logging.info("=" * 60)

    return log_file


# ======================
# UTIL
# ======================

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "meme"


def extract_and_save_image(response, out_path: Path) -> bool:
    """
    Robust extraction that matches your proven working version.
    Handles both as_image() and inline_data fallback.
    """
    logging.debug(f"Attempting to extract image to {out_path}")

    for cand_idx, cand in enumerate(response.candidates):
        logging.debug(f"  Candidate {cand_idx}: {len(cand.content.parts)} parts")
        for part_idx, part in enumerate(cand.content.parts):

            # Preferred SDK method
            try:
                img = part.as_image()
                if img:
                    img.save(out_path)
                    logging.debug(f"  ✓ Image extracted via as_image() method")
                    return True
            except Exception as e:
                logging.debug(f"  as_image() failed: {e}")
                pass

            # Fallback for inline image bytes
            if hasattr(part, "inline_data") and part.inline_data:
                with open(out_path, "wb") as f:
                    f.write(part.inline_data.data)
                logging.debug(f"  ✓ Image extracted via inline_data fallback")
                return True

    logging.warning(f"Failed to extract image from response")
    return False


def require_prompt(idea: dict, key: str) -> str:
    value = idea.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid prompt: {key}")
    return value


def force_image_prompt(prompt: str) -> str:
    """
    Re-introduces the hard requirement that made the old script reliable.
    """
    return (
        "OUTPUT REQUIREMENT:\n"
        "You MUST output an IMAGE.\n\n"
        + prompt.strip()
    )


# ======================
# MAIN
# ======================

def main():
    # Initialize logging first
    log_file = setup_logging()

    start_time = time.time()
    logging.info("Starting batch meme generation")
    logging.info(f"Model: {MODEL}")

    try:
        # API Key validation
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logging.error("GOOGLE_API_KEY environment variable not set")
            raise RuntimeError("Missing GOOGLE_API_KEY")
        logging.debug("API key loaded successfully")

        base_dir = Path(__file__).parent
        logging.debug(f"Working directory: {base_dir}")

        # Validate base image
        base_image_path = base_dir / BASE_IMAGE
        if not base_image_path.exists():
            logging.error(f"Base image not found: {base_image_path}")
            raise FileNotFoundError(f"Missing base image: {BASE_IMAGE}")
        logging.info(f"Base image loaded: {base_image_path}")

        base_image = Image.open(base_image_path)
        logging.debug(f"Base image size: {base_image.size}")

        # Handle IDEAS_FILE - support both absolute and relative paths
        ideas_path = Path(IDEAS_FILE)
        if not ideas_path.is_absolute():
            ideas_path = base_dir / ideas_path

        if not ideas_path.exists():
            logging.error(f"Ideas file not found: {ideas_path}")
            raise FileNotFoundError(f"Missing ideas file: {ideas_path}")

        logging.info(f"Loading ideas from: {ideas_path}")
        ideas = json.loads(ideas_path.read_text(encoding="utf-8"))
        items = ideas.get("items", [])

        if not items:
            logging.error("No items found in ideas file")
            raise RuntimeError("ideas.json contains no items")

        logging.info(f"Loaded {len(items)} meme ideas")

        # Handle OUT_DIR - support both absolute and relative paths
        out_dir = Path(OUT_DIR)
        if not out_dir.is_absolute():
            out_dir = base_dir / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {out_dir}")

        # Initialize API client
        logging.debug("Initializing Google Gemini API client")
        client = genai.Client(api_key=api_key)

        # 🔒 IMAGE-ONLY output — critical
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"]
        )
        logging.debug("API config: response_modalities=['IMAGE']")

        # Processing statistics
        stats = {
            "total": len(items),
            "successful": 0,
            "pass1_only": 0,
            "pass2_success": 0,
            "failed": 0,
            "timings": []
        }

        logging.info("=" * 60)
        logging.info("Starting meme generation")
        logging.info("=" * 60)

        for idx, idea in enumerate(items, start=1):
            idea_start = time.time()
            idea_id = idea.get("id", f"meme_{idx}")
            out_path = out_dir / f"{slugify(idea_id)}.png"

            logging.info(f"[{idx}/{len(items)}] Processing: {idea_id}")
            logging.debug(f"  Output path: {out_path}")

            try:
                reskin_prompt = force_image_prompt(
                    require_prompt(idea, "reskin_prompt")
                )
                annotation_prompt = force_image_prompt(
                    require_prompt(idea, "annotation_prompt")
                )
                logging.debug(f"  Prompts validated")

                # ------------------
                # PASS 1 — REQUIRED
                # ------------------
                logging.info("  → PASS 1 (reskin)")
                pass1_start = time.time()

                resp1 = client.models.generate_content(
                    model=MODEL,
                    contents=[reskin_prompt, base_image],
                    config=config,
                )

                pass1_duration = time.time() - pass1_start
                logging.debug(f"  PASS 1 API call completed in {pass1_duration:.2f}s")

                if not extract_and_save_image(resp1, out_path):
                    logging.error(f"  PASS 1 FAILED: No image returned for {idea_id}")
                    raise RuntimeError("PASS 1 FAILED: No image returned")

                logging.info(f"  ✓ PASS 1 completed in {pass1_duration:.2f}s")

                # ------------------
                # PASS 2 — BEST EFFORT
                # ------------------
                pass2_success = False

                for attempt in range(1, PASS2_RETRIES + 1):
                    try:
                        logging.info(
                            f"  → PASS 2 attempt {attempt}/{PASS2_RETRIES} "
                            f"(delaying {PASS2_DELAY_SECONDS}s)"
                        )
                        time.sleep(PASS2_DELAY_SECONDS)

                        pass2_start = time.time()
                        resp2 = client.models.generate_content(
                            model=MODEL,
                            contents=[annotation_prompt, Image.open(out_path)],
                            config=config,
                        )

                        pass2_duration = time.time() - pass2_start
                        logging.debug(f"  PASS 2 attempt {attempt} API call completed in {pass2_duration:.2f}s")

                        if extract_and_save_image(resp2, out_path):
                            pass2_success = True
                            logging.info(f"  ✓ PASS 2 succeeded on attempt {attempt} in {pass2_duration:.2f}s")
                            break

                    except Exception as e:
                        logging.warning(f"  PASS 2 attempt {attempt} failed: {e}")

                if not pass2_success:
                    logging.warning("  ⚠ PASS 2 failed all attempts — using PASS 1 image")
                    stats["pass1_only"] += 1
                else:
                    stats["pass2_success"] += 1

                idea_duration = time.time() - idea_start
                stats["timings"].append(idea_duration)
                logging.info(f"  ✓ Saved → {out_path} (total: {idea_duration:.2f}s)")
                stats["successful"] += 1

            except Exception as e:
                logging.error(f"  ✗ Failed to process {idea_id}: {e}", exc_info=True)
                stats["failed"] += 1

            logging.info("-" * 60)

        # Final summary
        total_duration = time.time() - start_time
        avg_time = sum(stats["timings"]) / len(stats["timings"]) if stats["timings"] else 0

        logging.info("=" * 60)
        logging.info("BATCH GENERATION COMPLETE")
        logging.info("=" * 60)
        logging.info(f"Total items:        {stats['total']}")
        logging.info(f"Successful:         {stats['successful']}")
        logging.info(f"  PASS 1 only:      {stats['pass1_only']}")
        logging.info(f"  PASS 2 success:   {stats['pass2_success']}")
        logging.info(f"Failed:             {stats['failed']}")
        logging.info(f"Total time:         {total_duration:.2f}s")
        logging.info(f"Average per meme:   {avg_time:.2f}s")
        logging.info(f"Log file:           {log_file}")
        logging.info("=" * 60)

        if stats["failed"] > 0:
            logging.warning(f"{stats['failed']} meme(s) failed to generate")
        else:
            logging.info("All memes generated successfully!")

    except Exception as e:
        logging.critical(f"FATAL ERROR: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
