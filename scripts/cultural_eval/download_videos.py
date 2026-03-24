import json
import subprocess
import pandas as pd
import time
import random
import os


# --- PART 1: DATA RESTRUCTURING & MAPPING ---

def restructure_and_map(cultural_source_data, framework_path):
    # Load the Evaluation Framework (Questions/Weights)
    with open(framework_path, 'r') as f:
        framework = json.load(f)

    final_list = []

    # Access the primary groups from your raw input
    actions_map = cultural_source_data.get("Action", {})
    scenes_map = cultural_source_data.get("Scene", {})
    objects_map = cultural_source_data.get("Object", {})

    # Iterate through 'Action' as the source of truth to build records
    for action_tag, items in actions_map.items():
        if not isinstance(items, list): continue

        for item in items:
            if not isinstance(item, dict): continue
            prompt = item.get("prompt", "")

            # Crash-proof tag lookup for Scene and Object
            scene_tag = "General"
            for s_tag, s_items in scenes_map.items():
                if isinstance(s_items, list):
                    if any(i.get("prompt") == prompt for i in s_items if isinstance(i, dict)):
                        scene_tag = s_tag
                        break

            object_tag = "General"
            for o_tag, o_items in objects_map.items():
                if isinstance(o_items, list):
                    if any(i.get("prompt") == prompt for i in o_items if isinstance(i, dict)):
                        object_tag = o_tag
                        break

            # Build the record structure
            record = {
                "id": item.get("id"),
                "prompt": prompt,
                "country": item.get("country"),
                "category": item.get("category"),
                "action_tag": action_tag,
                "scene_tag": scene_tag,
                "object_tag": object_tag,
                "action": [],
                "scene": [],
                "object": []
            }

            # Map Ground Truth from framework using prompt as the key
            if prompt in framework:
                gt = framework[prompt]
                record["action"] = gt.get("actions", [])
                record["scene"] = gt.get("scene", [])
                record["object"] = gt.get("objects", [])

            final_list.append(record)

    return final_list


# --- PART 2: YOUTUBE METADATA COLLECTION ---

def get_metadata_videos(query, tag_type):
    """Search YouTube for top 10 videos under 10 minutes."""
    search_string = f"{query} -stock -motion"

    cmd = [
        "yt-dlp",
        f"ytsearch10:{search_string}",
        "--match-filter", "duration < 600",
        "--flat-playlist",
        "--dump-single-json",
        "--quiet"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not result.stdout: return []

        data = json.loads(result.stdout)
        videos = []
        entries = data.get('entries', [])

        for entry in entries:
            if not entry: continue
            videos.append({
                'query_term': query,
                'tag_type': tag_type,
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


# --- PART 3: MAIN EXECUTION ---

# 1. Paths
# Assuming 'data' (your raw nested JSON) is already loaded into the script
framework_path = "/Users/anku/Desktop/greeting_ground_truth_experiments/youtube/india_evaluation_frameworks.json"
output_json_path = "cultural_data_final.json"

# 2. Process Cultural Data
print("📊 Restructuring cultural data and mapping ground truth...")
data = pd.read_json("cultural_data.json")
processed_records = restructure_and_map(data, framework_path)

# Save the JSON version for your records
with open(output_json_path, 'w') as f:
    json.dump(processed_records, f, indent=4)

# 3. Extract Unique Search Queries
query_map = {}
for rec in processed_records:
    if rec["action_tag"]: query_map[rec["action_tag"]] = "action_tag"
    if rec["scene_tag"]: query_map[rec["scene_tag"]] = "scene_tag"
    if rec["object_tag"]: query_map[rec["object_tag"]] = "object_tag"

all_queries = list(query_map.keys())
final_metadata_store = []

print(f"🚀 Starting YouTube collection for {len(all_queries)} unique tags...")

for i, q in enumerate(all_queries):
    tag_type = query_map[q]
    print(f"[{i + 1}/{len(all_queries)}] Searching {tag_type}: {q}")

    vids = get_metadata_videos(q, tag_type)
    final_metadata_store.extend(vids)

    # Adaptive sleep to avoid YouTube rate limits
    time.sleep(random.uniform(2, 4))

# 4. Save to CSV
df_results = pd.DataFrame(final_metadata_store)
if not df_results.empty:
    df_results['upload_date'] = pd.to_datetime(df_results['upload_date'], format='%Y%m%d', errors='coerce')
    output_csv = 'india_youtube_metadata_v2.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"✅ Success! Saved {len(df_results)} video records to {output_csv}.")
else:
    print("❌ No videos found.")