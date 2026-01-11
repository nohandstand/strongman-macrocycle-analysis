import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import whisper

VIDEOS_PATH = "data/raw/youtube_videos.parquet"
OUT_PATH = "data/raw/whisper_transcripts.parquet"
AUDIO_DIR = Path("data/audio")

# Start small for testing; raise later
MAX_VIDEOS = 10

# Whisper model sizes: tiny, base, small, medium, large
WHISPER_MODEL = "base"


def download_audio(video_id: str) -> Path:
    """
    Downloads best audio as .mp3 into data/audio/
    Returns the output filepath.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    outtmpl = str(AUDIO_DIR / f"{video_id}.%(ext)s")

    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "-o", outtmpl,
        url,
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    mp3_path = AUDIO_DIR / f"{video_id}.mp3"
    if not mp3_path.exists():
        # yt-dlp sometimes uses m4a depending on settings; fallback check
        for ext in ("m4a", "webm", "opus"):
            p = AUDIO_DIR / f"{video_id}.{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Audio not found for {video_id} after download.")
    return mp3_path


def main():
    if not os.path.exists(VIDEOS_PATH):
        raise RuntimeError("Missing video metadata parquet. Run src/01_pull_youtube.py first.")

    vids = pd.read_parquet(VIDEOS_PATH)
    video_ids = vids["video_id"].dropna().astype(str).unique().tolist()

    print(f"Total videos: {len(video_ids)}")
    video_ids = video_ids[:MAX_VIDEOS]
    print(f"Transcribing first {len(video_ids)} videos for a test run...")

    model = whisper.load_model(WHISPER_MODEL)

    existing = None
    done = set()
    if os.path.exists(OUT_PATH):
        existing = pd.read_parquet(OUT_PATH)
        done = set(existing["video_id"].astype(str).tolist())
        print(f"Resuming: {len(done)} already in {OUT_PATH}")

    rows = []
    for i, vid in enumerate(video_ids, start=1):
        if vid in done:
            continue

        fetched_at = datetime.now(timezone.utc)

        try:
            audio_path = download_audio(vid)
            result = model.transcribe(str(audio_path))
            text = (result.get("text") or "").strip()
            lang = result.get("language")

            rows.append({
                "video_id": vid,
                "has_transcript": True if text else False,
                "language_code": lang,
                "source": f"whisper_{WHISPER_MODEL}",
                "transcript_text": text if text else None,
                "error_type": None,
                "error_message": None,
                "fetched_at_utc": fetched_at,
            })
            print(f"[{i}/{len(video_ids)}] ok: {vid} ({len(text)} chars)")

        except Exception as e:
            rows.append({
                "video_id": vid,
                "has_transcript": False,
                "language_code": None,
                "source": f"whisper_{WHISPER_MODEL}",
                "transcript_text": None,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "fetched_at_utc": fetched_at,
            })
            print(f"[{i}/{len(video_ids)}] fail: {vid} -> {type(e).__name__}")

        # checkpoint every 5
        if len(rows) > 0 and len(rows) % 5 == 0:
            out_df = pd.DataFrame(rows)
            if existing is not None:
                out_df = pd.concat([existing, out_df], ignore_index=True)
            out_df.to_parquet(OUT_PATH, index=False)
            print(f"Checkpoint saved: {OUT_PATH} ({len(out_df)} rows)")

    out_df = pd.DataFrame(rows)
    if existing is not None:
        out_df = pd.concat([existing, out_df], ignore_index=True)
    os.makedirs("data/raw", exist_ok=True)
    out_df.to_parquet(OUT_PATH, index=False)

    ok = int(out_df["has_transcript"].sum()) if len(out_df) else 0
    print(f"Saved: {OUT_PATH}")
    print(f"Transcripts found: {ok}/{len(out_df)}")


if __name__ == "__main__":
    main()
