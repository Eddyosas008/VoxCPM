"""Narrate a whole book / long script to per-chapter WAV files — robustly.

Designed for long-form content (books, guided meditations, podcast scripts) where
a single generation call is not possible (the engine errors above ~8192 tokens)
and holding the whole audio in memory is wasteful.

How it works
------------
- Reads a UTF-8 ``.txt`` file. Chapters are separated by a line containing only
  ``---`` (Markdown horizontal rule) by default, or by ``--chapter-regex``. If no
  separator is found, the whole text is treated as a single chapter.
- Each chapter is split into sentence chunks (reusing ``app._split_text_into_chunks``)
  so every call stays well under the engine's token limit.
- Chunks are synthesized with the SAME seed for a consistent voice, then stitched
  per chapter with a short silence.
- **Memory-safe:** only one chapter is held in memory at a time, never the whole book.
- **Resumable:** a chapter whose output ``.wav`` already exists is skipped, so an
  interrupted run continues where it left off. Use ``--force`` to regenerate.
- The denoiser is never loaded (narration uses no reference audio), so startup is
  fast and does not touch ModelScope.

Examples
--------
  # Preview segmentation without generating anything (fast, no model load):
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --voice "Narrateur profond & calme" --dry-run

  # Narrate with a preset voice:
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --voice "Narrateur profond & calme"

  # Narrate with a custom voice (description + seed):
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --description "Voix ..." --seed 123

  # On a CUDA GPU (far faster):
  ./.venv/Scripts/python.exe scripts/narrate_book.py livre.txt --voice "..." --device cuda
"""
import argparse
import re
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Make the repo root importable so we can reuse app.py's helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def split_chapters(text: str, chapter_regex: str | None) -> list[str]:
    """Split the book text into chapters. Defaults to Markdown '---' rules."""
    pattern = chapter_regex if chapter_regex else r"(?m)^\s*---\s*$"
    parts = re.split(pattern, text)
    chapters = [p.strip() for p in parts if p.strip()]
    return chapters or [text.strip()]


def resolve_voice(args) -> tuple[str, int | None]:
    """Return (description, seed) from a preset name or explicit --description/--seed."""
    if args.voice:
        preset = app._PRESET_BY_NAME.get(args.voice)
        if preset is None:
            names = ", ".join(repr(v["name"]) for v in app.PRESET_VOICES)
            raise SystemExit(f"Unknown voice {args.voice!r}. Available presets: {names}")
        return preset["description"], preset["seed"]
    return (args.description or ""), args.seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the .txt file to narrate")
    parser.add_argument("--voice", help="Preset voice name (see conf/preset_voices.json)")
    parser.add_argument("--description", help="Custom voice description (if not using --voice)")
    parser.add_argument("--seed", type=int, help="Seed for the custom voice (fixes the voice identity)")
    parser.add_argument("--outdir", help="Output directory (default: output/book_<filename>)")
    parser.add_argument("--device", default="cpu", help="auto, cpu, mps, cuda, or cuda:N (default: cpu)")
    parser.add_argument("--model-id", default="openbmb/VoxCPM2", help="Model path or HF repo id")
    parser.add_argument("--chunk-max-chars", type=int, default=app._CHUNK_MAX_CHARS,
                        help=f"Max characters per chunk (default: {app._CHUNK_MAX_CHARS})")
    parser.add_argument("--silence", type=float, default=app._CHUNK_SILENCE_SEC,
                        help=f"Silence between chunks in seconds (default: {app._CHUNK_SILENCE_SEC})")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG guidance scale (default: 2.0)")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps (default: 10)")
    parser.add_argument("--no-normalize", action="store_true", help="Disable text normalization")
    parser.add_argument("--chapter-regex", help="Regex (MULTILINE) that separates chapters (default: '^---$')")
    parser.add_argument("--force", action="store_true", help="Regenerate chapters even if their .wav exists")
    parser.add_argument("--dry-run", action="store_true", help="Show the segmentation plan, generate nothing")
    parser.add_argument("--continuity", action="store_true",
                        help="EXPERIMENTAL: chain each chunk from the previous one (prompt-cache "
                             "continuation) for smoother joins, instead of same-seed only. Slower; "
                             "resets at each chapter boundary. Tune on a GPU (slow to iterate on CPU).")
    args = parser.parse_args()

    if not args.voice and not args.description:
        raise SystemExit("Provide either --voice <preset name> or --description <text> [--seed N].")

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")
    text = in_path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Input file is empty: {in_path}")

    description, seed = resolve_voice(args)
    chapters = split_chapters(text, args.chapter_regex)
    outdir = Path(args.outdir) if args.outdir else app._OUTPUT_DIR / f"book_{app._sanitize_filename(in_path.stem)}"

    # Plan: chunk every chapter up front so --dry-run can show the full picture.
    plan = [(i, ch, app._split_text_into_chunks(ch, args.chunk_max_chars)) for i, ch in enumerate(chapters, 1)]
    total_chunks = sum(len(chunks) for _, _, chunks in plan)
    total_chars = sum(len(ch) for ch in chapters)
    print(f"Input      : {in_path}")
    print(f"Voice      : {args.voice or '(custom)'} | seed={seed}")
    print(f"Chapters   : {len(chapters)} | chunks: {total_chunks} | chars: {total_chars}")
    print(f"Output dir : {outdir}")
    for i, _, chunks in plan:
        print(f"  chapter {i:03d}: {len(chunks)} chunk(s)")

    if args.dry_run:
        print("\nDry run — nothing generated.")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    demo = app.VoxCPMDemo(model_id=args.model_id, device=args.device, load_denoiser=False)
    normalize = not args.no_normalize

    started = time.strftime("%H:%M:%S")
    print(f"\nStarting narration at {started} (device={args.device}). This is slow on CPU.\n", flush=True)

    for i, _, chunks in plan:
        out = outdir / f"chapitre_{i:03d}.wav"
        if out.is_file() and not args.force:
            print(f"[chapter {i:03d}/{len(plan)}] exists, skipping -> {out.name}", flush=True)
            continue
        print(f"[chapter {i:03d}/{len(plan)}] {len(chunks)} chunk(s) ...", flush=True)
        parts: list[np.ndarray] = []
        sr = None
        # Continuity: chain each chunk from the immediately previous one only
        # (bounded window → never overflows the KV cache). Reset per chapter.
        prev_wav_path: str | None = None
        prev_text: str | None = None
        tmp_paths: list[str] = []
        try:
            for j, chunk in enumerate(chunks):
                if args.continuity and prev_wav_path is not None:
                    # Voice comes from the running audio, so drop the control text.
                    sr, wav, _ = demo.generate_tts_audio(
                        text_input=chunk,
                        control_instruction="",
                        reference_wav_path_input=prev_wav_path,
                        prompt_text=prev_text,
                        cfg_value_input=args.cfg,
                        do_normalize=normalize,
                        inference_timesteps=args.steps,
                        seed=seed,
                    )
                else:
                    sr, wav, _ = demo.generate_tts_audio(
                        text_input=chunk,
                        control_instruction=description,
                        cfg_value_input=args.cfg,
                        do_normalize=normalize,
                        inference_timesteps=args.steps,
                        seed=seed,
                    )
                if j > 0:
                    parts.append(np.zeros(int(sr * args.silence), dtype=wav.dtype))
                parts.append(wav)
                if args.continuity:  # stash this chunk as the prompt for the next one
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp_paths.append(tmp.name)
                    sf.write(tmp_paths[-1], wav, sr)
                    prev_wav_path, prev_text = tmp_paths[-1], chunk
                print(f"    chunk {j + 1}/{len(chunks)} done", flush=True)
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        book = np.concatenate(parts)
        sf.write(str(out), book, sr)
        print(f"[chapter {i:03d}/{len(plan)}] saved -> {out.name} ({len(book) / sr:.1f}s)", flush=True)

    print(f"\nDone. Chapter files are in: {outdir}", flush=True)
    print("Tip: concatenate them into one file with your audio tool, e.g. ffmpeg concat.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
