import os
import time
from datetime import datetime

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    TooManyRequests,
)

RAW_VIDEOS_PATH = "data/raw/youtube_videos.parquet"
OUT_PATH = "data/raw/youtube_transcripts.parquet"


def pick_transcript(transcript_list, preferred_langs=("en",)):
    """
    Strategy:
      1) manual transcript in preferred language
      2) auto transcript in preferred language
      3) any manual transcript
      4) any auto transcript
    """
    for lang in preferred_langs:
        try:
            t = transcript_list.find_manually_created_transcript([lang])
            return t, "manual"
        except Exception:
            pass

    for lang in preferred_langs:
        try:
            t = transcript_list.find_generated_transcript([lang])
            return t, "auto"
        except Exception:
            pass

    try:
        manuals = [t for t in transcript_list if not t.is_generated]
        if manuals:
            return manuals[0], "manual"
    except Exception:
        pass

    try:
        autos = [t for t in transcript_list if t.is_generated]
        if autos:
            return autos[0], "auto"
    except Exception:
        pass

    return None, None


def fetch_one(video_id: str, preferred_langs=("en",), sleep_s=0.15) -> dict:
    formatter = TextFormatter()
    fetched_at = datetime.utcnow()

    try:
        tlist = YouTubeTranscriptApi.list_transcripts(video_id)
        chosen, source = pick_transcript(tlist, preferred_langs=preferred_langs)

        if chosen is None:
            return {
                "video_id": video_id,
                "has_transcript": False,
                "language_code": None,
                "is_generated": None,
                "source": None,
                "transcript_text": None,
                "error_type": "NoTranscriptFound",
                "error_message": "No usable transcript found",
                "fetched_at_utc": fetched_at,
            }

        segments = chosen.fetch()
        text = formatter.format_transcript(segments)

        time.sleep(sleep_s)

        return {
            "video_id": video_id,
            "has_transcript": True,
            "language_code": chosen.language_code,
            "is_generated": bool(getattr(chosen, "is_generated", None)),
            "source": source,
            "transcript_text": text,
            "error_type": None,
            "error_message": None,
            "fetched_at_utc": fetched_at,
        }

    except TooManyRequests as e:
        return {
            "video_id": video_id,
            "has_transcript": False,
            "language_code": None,
            "is_generated": None,
            "source": None,
            "transcript_text": None,
            "error_type": "TooManyRequests",
            "error_message": str(e),
            "fetched_at_utc": fetched_at,
        }
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return {
            "video_id": video_id,
            "has_transcript": False,
            "language_code": None,
            "is_generated": None,
            "source": None,
            "transcript_text": None,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "fetched_at_utc": fetched_at,
        }
    except Exception as e:
        return {
            "video_id": video_id,
            "has_transcript": False,
            "language_code": None,
            "is_generated": None,
            "source": None,
            "transcript_text": None,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "fetched_at_utc": fetched_at,
        }


def main():
    if not os.path.exists(RAW_VIDEOS_PATH):
        raise RuntimeError(
            f"Missing {RAW_VIDEOS_PATH}. Run src/01_pull_youtube.py first."
        )

    videos = pd.read_parquet(RAW_VIDEOS_PATH)
    video_ids = videos["video_id"].dropna().astype(str).unique().tolist()
    print(f"Loaded {len(video_ids)} video ids from {RAW_VIDEOS_PATH}")

    existing = None
    done_ids = set()
    if os.path.exists(OUT_PATH):
        existing = pd.read_parquet(OUT_PATH)
        done_ids = set(existing["video_id"].astype(str).tolist())
        print(f"Found existing {OUT_PATH} with {len(done_ids)} rows. Resuming...")

    rows = []
    total = len(video_ids)

    for idx, vid in enumerate(video_ids, start=1):
        if vid in done_ids:
            continue

        rec = fetch_one(vid, preferred_langs=("en",), sleep_s=0.15)
        rows.append(rec)

        if idx % 25 == 0:
            ok = sum(1 for r in rows if r["has_transcript"])
            print(f"[{idx}/{total}] new={len(rows)} ok={ok}")

        if rec["error_type"] == "TooManyRequests":
            print("Rate limited (TooManyRequests). Sleeping 60s then continuing...")
            time.sleep(60)

        if len(rows) > 0 and len(rows) % 100 == 0:
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

    ok_total = int(out_df["has_transcript"].sum()) if len(out_df) else 0
    print(f"Saved: {OUT_PATH}")
    print(f"Transcripts found: {ok_total}/{len(out_df)}")

    if len(out_df) > 0:
        print("\nTop error types:")
        print(out_df["error_type"].fillna("None").value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
