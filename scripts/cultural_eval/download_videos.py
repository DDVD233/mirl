"""
Download YouTube videos referenced in all_evaluation_frameworks.json.
- Deduplicates: each video downloaded once by video ID
- Resumable: skips already-downloaded videos
- Updates JSON with local_path field in each video object
- Saves to: videos/{country}/{video_id}.mp4
"""

import json
import os
import subprocess
import time

INPUT_FILE   = "all_evaluation_frameworks.json"
DOWNLOAD_DIR = "videos"
DELAY        = 1.0  # seconds between downloads

TAG_FIELDS = [
    "object_tag_youtube_video",
    "action_tag_youtube_video",
    "scene_tag_youtube_video",
]


def get_video_id(url):
    return url.split("watch?v=")[-1]


def collect_videos(data):
    """Return dict of {video_id: {url, country, ...}} — deduplicated."""
    videos = {}
    for entry in data.values():
        country = entry.get("country", "unknown").replace("_", "-")
        for field in TAG_FIELDS:
            for v in entry.get(field, []):
                url = v.get("video_url")
                if not url:
                    continue
                vid_id = get_video_id(url)
                if vid_id not in videos:
                    videos[vid_id] = {"url": url, "country": country}
    return videos


def expected_path(country, video_id):
    return os.path.join(DOWNLOAD_DIR, country, f"{video_id}.mp4")


def cleanup_partial(out_dir, video_id):
    """Remove .part and intermediate files for a video ID."""
    if not os.path.isdir(out_dir):
        return
    for f in os.listdir(out_dir):
        if f.startswith(video_id) and (".part" in f or ".f1" in f or ".f2" in f or ".f3" in f):
            os.remove(os.path.join(out_dir, f))


def download_video(url, country, video_id):
    out_dir = os.path.join(DOWNLOAD_DIR, country)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}.%(ext)s")

    # Clean up any leftover .part files from previous attempts
    cleanup_partial(out_dir, video_id)

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        final_path = os.path.join(out_dir, f"{video_id}.mp4")
        if os.path.exists(final_path):
            return final_path
        # Download failed or left partial files — clean up
        cleanup_partial(out_dir, video_id)
        return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {url}")
        cleanup_partial(out_dir, video_id)
        return None
    except Exception as e:
        print(f"  Error: {e}")
        cleanup_partial(out_dir, video_id)
        return None


def update_json_paths(data, path_map):
    """Add local_path to each video object in the JSON."""
    for entry in data.values():
        for field in TAG_FIELDS:
            for v in entry.get(field, []):
                url = v.get("video_url")
                if url:
                    vid_id = get_video_id(url)
                    if vid_id in path_map:
                        v["local_path"] = path_map[vid_id]


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    all_videos = collect_videos(data)
    total = len(all_videos)
    print(f"Total unique videos: {total}")

    # Check already downloaded
    path_map = {}
    to_download = []
    for vid_id, info in all_videos.items():
        path = expected_path(info["country"], vid_id)
        # Also check if already downloaded (any extension)
        country_dir = os.path.join(DOWNLOAD_DIR, info["country"])
        found = None
        if os.path.isdir(country_dir):
            for f in os.listdir(country_dir):
                if f.startswith(vid_id) and f.endswith(".mp4") and ".part" not in f:
                    found = os.path.join(country_dir, f)
                    break
        if found:
            path_map[vid_id] = found
        else:
            to_download.append((vid_id, info))

    print(f"Already downloaded: {len(path_map)}")
    print(f"To download: {len(to_download)}")
    print()

    for i, (vid_id, info) in enumerate(to_download[2499:], 2500):
        print(f"[{i}/{len(to_download)}] {vid_id} ({info['country']})")
        path = download_video(info["url"], info["country"], vid_id)
        if path:
            path_map[vid_id] = path
            print(f"  Saved: {path}")
        else:
            print(f"  Failed: {info['url']}")
        time.sleep(DELAY)

        # Save JSON every 100 downloads
        if i % 100 == 0:
            update_json_paths(data, path_map)
            with open(INPUT_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  >> JSON updated ({i}/{len(to_download)})")

    # Final JSON update
    update_json_paths(data, path_map)
    with open(INPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone! Downloaded {len(path_map)}/{total} videos.")
    failed = total - len(path_map)
    if failed:
        print(f"Failed/unavailable: {failed}")


if __name__ == "__main__":
    main()