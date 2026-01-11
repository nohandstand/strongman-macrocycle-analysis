import os
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from tqdm import tqdm


def iso8601_duration_to_seconds(duration: str) -> int | None:
    """Convert ISO8601 duration like 'PT1H2M3S' to seconds."""
    if not duration or not isinstance(duration, str):
        return None
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", duration)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mins * 60 + s


def get_channel_id_from_query(youtube, query: str) -> str:
    """
    Resolve a channel using YouTube search. Works with handles like '@HattonStrength'
    or plain text like 'Lucas Hatton strength'.
    """
    resp = youtube.search().list(
        part="snippet",
        q=query,
        type="channel",
        maxResults=5,
    ).execute()

    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"No channel found for query: {query}")

    # First result
    return items[0]["snippet"]["channelId"]


def get_uploads_playlist_id(youtube, channel_id: str) -> str:
    resp = youtube.channels().list(
        part="contentDetails,snippet",
        id=channel_id,
        maxResults=1,
    ).execute()

    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"Channel not found: {channel_id}")

    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_playlist_video_ids(youtube, playlist_id: str) -> list[str]:
    ids: list[str] = []
    next_page_token = None

    while True:
        resp = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        ).execute()

        for it in resp.get("items", []):
            ids.append(it["contentDetails"]["videoId"])

        next_page_token = resp.get("nextPageToken")
        if not next_page_token:
            break

    return ids


def fetch_video_details(youtube, video_ids: list[str]) -> pd.DataFrame:
    rows = []
    for i in tqdm(range(0, len(video_ids), 50), desc="Fetching video details"):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(batch),
            maxResults=50,
        ).execute()

        for it in resp.get("items", []):
            snippet = it.get("snippet", {})
            content = it.get("contentDetails", {})
            stats = it.get("statistics", {})

            published_at = snippet.get("publishedAt")
            published_dt = (
                datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                if published_at
                else None
            )

            duration = content.get("duration")
            duration_seconds = iso8601_duration_to_seconds(duration)

            vid = it.get("id")
            rows.append(
                {
                    "video_id": vid,
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": published_dt,
                    "channel_id": snippet.get("channelId"),
                    "channel_title": snippet.get("channelTitle"),
                    "duration_iso8601": duration,
                    "duration_seconds": duration_seconds,
                    "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
                    "like_count": int(stats["likeCount"]) if "likeCount" in stats else None,
                    "comment_count": int(stats["commentCount"]) if "commentCount" in stats else None,
                    "url": f"https://www.youtube.com/watch?v={vid}" if vid else None,
                }
            )

    df = pd.DataFrame(rows).sort_values("published_at").reset_index(drop=True)
    return df


def main():
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing YOUTUBE_API_KEY in .env")

    youtube = build("youtube", "v3", developerKey=api_key)

    # ====== CONFIG (EDIT THIS) ======
    # Option A: Use a handle/query (easy, but can pick wrong channel if ambiguous)
    channel_query = "@HattonStrength"

    # Option B: If you know the exact channel ID, paste it here and set use_channel_id=True
    use_channel_id = False
    channel_id = "PASTE_CHANNEL_ID_HERE_IF_USING_OPTION_B"
    # =================================

    if not use_channel_id:
        channel_id = get_channel_id_from_query(youtube, channel_query)

    uploads_playlist_id = get_uploads_playlist_id(youtube, channel_id)
    video_ids = list_playlist_video_ids(youtube, uploads_playlist_id)

    print(f"Channel ID: {channel_id}")
    print(f"Found {len(video_ids)} videos in uploads playlist.")

    df = fetch_video_details(youtube, video_ids)

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/youtube_videos.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df.tail(5)[["published_at", "title", "url"]].to_string(index=False))


if __name__ == "__main__":
    main()

