#!/usr/bin/env python3
"""
pdf2audio.py — Convert PDF files to Czech audio (MP3) using PyMuPDF + edge-tts.

Usage:
    # Single file
    python pdf2audio.py notes.pdf

    # Multiple files
    python pdf2audio.py notes1.pdf notes2.pdf

    # Whole directory
    python pdf2audio.py ./study_materials/

    # Options
    python pdf2audio.py --voice cs-CZ-AntoninNeural --rate "-10%" --output-dir ./audio notes.pdf
"""

from __future__ import annotations

import argparse
import asyncio
from pydoc import text
import re
import sys
import textwrap
from pathlib import Path
import requests
import json
import os
import logging

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: PDF → raw text  (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using PyMuPDF, preserving page order."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")  # plain-text extraction
        if text.strip():
            pages.append(text)
    doc.close()

    if not pages:
        raise ValueError(f"No text could be extracted from {pdf_path}")

    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# Step 2: Text cleanup
# ---------------------------------------------------------------------------

API_KEY = os.getenv("OPENROUTER_API_KEY")  # Set
MAX_API_CHUNK_CHARS = 5000
TOTAL_API_COST = 0.0

def _get_chunk_summary(chunk_text: str) -> tuple[str, float]:
    """
    Get a very brief summary of the chunk via API.
    """
    if not API_KEY:
        return "", 0.0

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-username/study-helper",
    }
    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": [
            {"role": "system", "content": "Provide a single, extremely brief sentence in Czech summarizing the core topics of the provided text. Focus only on high-level subjects (e.g., 'Discussed hypertension symptoms and diagnosis')."},
            {"role": "user", "content": chunk_text}
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        cost = usage.get("cost", 0.0)
        summary = data['choices'][0]['message']['content'].strip()
        return summary, cost
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return "(Summary unavailable)", 0.0


def _call_api_for_clean_text(raw_text: str, chunk_index: int, total_chunks: int, history_summary: str) -> tuple[str, float]:
    """
    Calls OpenRouter API to refine a single chunk of text with context.
    Returns (refined_text, cost).
    """
    if not API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. Skipping API refinement.")
        return raw_text, 0.0

    url = "https://openrouter.ai/api/v1/chat/completions"
    
    context_msg = f"This is part {chunk_index} of {total_chunks}."
    if history_summary:
        context_msg += f"\n\nContext of what was discussed in previous parts:\n{history_summary}"

    system_prompt = (
        "You are an expert tutor preparing audio-learning materials. "
        "Rewrite the following raw study notes into a natural, fluid, spoken-word script in Czech. "
        "Strictly follow these rules:\n"
        "1. Convert all bullet points/lists into coherent, complete sentences.\n"
        "2. Keep medical terminology accurate, but explain abbreviations if necessary for clarity.\n"
        "3. Use a helpful, educational, and steady tone suitable for listening.\n"
        "4. Remove visual markers like hyphens, bullets, or 'o' characters.\n"
        "5. If there are clearly structured headers, use them to create smooth transitions.\n"
        "6. Do not output anything other than the final script text."
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-username/study-helper",
    }

    user_content = f"{context_msg}\n\nStudy notes to rewrite:\n\n{raw_text}"

    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = usage.get("cost")
        
        if cost is None:
            input_cost = (prompt_tokens / 1_000_000) * 0.075
            output_cost = (completion_tokens / 1_000_000) * 0.30
            cost = input_cost + output_cost

        logger.info(f"OpenRouter rewrite chunk {chunk_index}: {prompt_tokens} prompt + {completion_tokens} completion tokens. Cost: ${cost:.6f}")

        text = data['choices'][0]['message']['content'].strip()
        text = text.replace("*", "")
        return text, cost
    except Exception as e:
        logger.error(f"Error calling OpenRouter API: {e}")
        return raw_text, 0.0


def clean_text(raw_text: str, debug_dir: Path | None = None) -> str:
    """
    Wrapper for clean_text that splits input into parts and merges them into
    chunks suitable for API calls, maintaining historical context.
    """
    global TOTAL_API_COST
    
    # Split by two empty lines
    parts = re.split(r"\n\n+", raw_text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if current_length + len(part) + 2 > MAX_API_CHUNK_CHARS and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(part)
        current_length += len(part) + 2
        
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    refined_parts = []
    history_summaries = []
    
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        if debug_dir:
            (debug_dir / f"chunk_{i:03d}_raw.txt").write_text(chunk, encoding="utf-8")

        current_history = "\n".join(history_summaries)
        logger.info(f"Processing API chunk {i}/{total_chunks}...")
        
        # 1. Rewrite the text with context
        refined_text, rewrite_cost = _call_api_for_clean_text(chunk, i, total_chunks, current_history)
        refined_parts.append(refined_text)
        TOTAL_API_COST += rewrite_cost
        
        if debug_dir:
            (debug_dir / f"chunk_{i:03d}_cleaned.txt").write_text(refined_text, encoding="utf-8")

        # 2. Generate summary for future context (except for the last chunk)
        if i < total_chunks:
            logger.info(f"Generating summary for chunk {i} context...")
            summary, summary_cost = _get_chunk_summary(refined_text)
            history_summaries.append(f"- Part {i}: {summary}")
            TOTAL_API_COST += summary_cost

    return "\n\n".join(refined_parts)


# ---------------------------------------------------------------------------
# Step 3: Text → Audio  (edge-tts)
# ---------------------------------------------------------------------------

# edge-tts has a practical per-request text limit (~roughly a few thousand chars).
# We split into chunks to avoid issues and to get more reliable output.
MAX_CHUNK_CHARS = 3000


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text into chunks at sentence boundaries ('. ') so each chunk
    is at most max_chars long.
    """
    sentences = re.split(r"(?<=\.)\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if current_len + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


async def synthesize_to_mp3(
    text: str,
    output_path: Path,
    voice: str = "cs-CZ-VlastaNeural",
    rate: str = "+0%",
) -> None:
    """
    Convert text to MP3 using edge-tts.

    Splits long text into chunks, synthesizes each, and concatenates
    the resulting MP3 data into a single file.
    """
    import edge_tts

    chunks = split_into_chunks(text)
    total = len(chunks)

    with open(output_path, "wb") as out_file:
        for i, chunk in enumerate(chunks, start=1):
            print(f"    Synthesizing chunk {i}/{total} ({len(chunk)} chars)...")
            communicate = edge_tts.Communicate(chunk, voice, rate=rate)
            async for message in communicate.stream():
                if message["type"] == "audio":
                    out_file.write(message["data"])

    print(f"    ✓ Saved: {output_path}  ({output_path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    voice: str,
    rate: str,
) -> None:
    """Full pipeline for one PDF file."""
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")

    # Step 1: Extract
    print("  [1/3] Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"         Extracted {len(raw_text)} characters from {pdf_path.name}")

    txt_path = output_dir / pdf_path.with_suffix(".extracted.txt").name
    txt_path.write_text(raw_text, encoding="utf-8")

    # Step 2: Clean
    print("  [2/3] Cleaning text...")
    
    # Create debug directory for chunks
    debug_dir = output_dir / f"{pdf_path.stem}_chunks"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    clean = clean_text(raw_text, debug_dir=debug_dir)
    print(f"         Cleaned text: {len(clean)} characters")

    # Optionally save intermediate text for inspection / LLM rewriting
    txt_path = output_dir / pdf_path.with_suffix(".txt").name
    txt_path.write_text(clean, encoding="utf-8")
    print(f"         Intermediate text saved to: {txt_path}")

    # Step 3: Synthesize
    print("  [3/3] Synthesizing audio...")
    mp3_path = output_dir / pdf_path.with_suffix(".mp3").name
    asyncio.run(synthesize_to_mp3(clean, mp3_path, voice=voice, rate=rate))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF study notes to Czech audio files (MP3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python pdf2audio.py notes.pdf
              python pdf2audio.py *.pdf
              python pdf2audio.py ./pdfs/ --voice cs-CZ-AntoninNeural
              python pdf2audio.py notes.pdf --rate "-15%"  # slower speech
        """),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="PDF file(s) or director(ies) containing PDFs.",
    )
    parser.add_argument(
        "--voice",
        default="cs-CZ-VlastaNeural",
        choices=["cs-CZ-VlastaNeural", "cs-CZ-AntoninNeural"],
        help="Czech TTS voice (default: VlastaNeural = female).",
    )
    parser.add_argument(
        "--rate",
        default="+0%",
        help='Speech rate, e.g. "+20%%" for faster or "-15%%" for slower (default: +0%%).',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MP3 files (default: same as each PDF).",
    )

    args = parser.parse_args()

    # Collect all PDF paths
    pdf_files: list[Path] = []
    for inp in args.inputs:
        if inp.is_dir():
            found = sorted(inp.glob("*.pdf"))
            if not found:
                print(f"Warning: no PDFs found in {inp}", file=sys.stderr)
            pdf_files.extend(found)
        elif inp.is_file() and inp.suffix.lower() == ".pdf":
            pdf_files.append(inp)
        else:
            print(f"Warning: skipping {inp} (not a PDF or directory)", file=sys.stderr)

    if not pdf_files:
        print("Error: no PDF files to process.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF file(s) to convert.")

    for pdf_path in pdf_files:
        out_dir = args.output_dir or pdf_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            process_single_pdf(pdf_path, out_dir, args.voice, args.rate)
        except Exception as e:
            print(f"  ✗ Error processing {pdf_path.name}: {e}", file=sys.stderr)

    print(f"\n{'='*60}")
    print(f"Done! Processed {len(pdf_files)} file(s).")
    print(f"Total OpenRouter API cost: ${TOTAL_API_COST:.6f}")


if __name__ == "__main__":
    main()