import pandas as pd
import subprocess
import json
import time
import random
import os

# 1. Load the Cultural Data to extract Action, Scene, and Object tags
cultural_data_path = "cultural_data.json"

with open(cultural_data_path, 'r') as f:
    cultural_data = json.load(f)

# Extract unique keys from each category
actions = list(cultural_data.get('Action', {}).keys())
scenes = list(cultural_data.get('Scene', {}).keys())
objects = list(cultural_data.get('Object', {}).keys())

# Create a mapping so we know if a query was an action_tag, scene_tag, or object_tag
query_map = {}
for a in actions: query_map[a] = "action_tag"
for s in scenes: query_map[s] = "scene_tag"
for o in objects: query_map[o] = "object_tag"

# Master list of unique queries
all_queries = list(query_map.keys())


def get_metadata_videos(query, tag_type):
    """
    Search YouTube for top 10 videos under 10 minutes for a specific query.
    """
    # Filter for duration < 600s (10 mins)
    search_string = f"{query} -stock -motion"

    cmd = [
        "yt-dlp",
        f"ytsearch10:{search_string}",
        "--match-filter", "duration < 600",  # Changed from 60 to 600
        "--flat-playlist",
        "--dump-single-json",
        "--quiet"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not result.stdout:
            return []

        data = json.loads(result.stdout)
        videos = []
        entries = data.get('entries', [])

        for entry in entries:
            if not entry: continue
            videos.append({
                'query_term': query,
                'tag_type': tag_type,  # Stores if it was action_tag, scene_tag, etc.
                'video_title': entry.get('title'),
                'video_url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                'channel_name': entry.get('uploader'),
                'view_count': entry.get('view_count'),
                'upload_date': entry.get('upload_date'),
                'duration': entry.get('duration'),
                'description': entry.get('description')[:300] if entry.get('description') else ""
            })
        return videos
    except Exception as e:
        print(f"Error for '{query}': {e}")
        return []


final_metadata_store = []

print(f"🚀 Starting Collection for {len(all_queries)} unique tags...")

for i, q in enumerate(all_queries):
    tag_type = query_map[q]
    print(f"[{i + 1}/{len(all_queries)}] Searching {tag_type}: {q}")

    vids = get_metadata_videos(q, tag_type)
    final_metadata_store.extend(vids)

    # Adaptive sleep to avoid YouTube rate limits
    time.sleep(random.uniform(1.5, 3))

# 2. Process and Save
df_results = pd.DataFrame(final_metadata_store)

if not df_results.empty:
    # Convert dates
    df_results['upload_date'] = pd.to_datetime(df_results['upload_date'], format='%Y%m%d', errors='coerce')

    # Save results
    output_file = 'india_youtube_metadata_v2.csv'
    df_results.to_csv(output_file, index=False)
    print(f"✅ Success! Collected {len(df_results)} video records and saved to {output_file}.")
else:
    print("❌ No videos found. Check your yt-dlp installation or connection.")