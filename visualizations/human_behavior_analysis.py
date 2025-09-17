import json
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from collections import Counter
import numpy as np
import subprocess
from tqdm import tqdm

# Define color palette
colors = ["#da81c1", "#7dbfa7", "#b0d766", "#8ca0cb", "#ee946c", "#da81c1"]

# Path to the annotations file
annotations_path = Path("/Users/dvd/Downloads/human_behaviour_data/all_annotations.jsonl")
metadata_cache_path = Path("/Users/dvd/Downloads/human_behaviour_data/media_metadata_cache.json")

def load_annotations():
    """Load all annotations from JSONL file"""
    annotations = []
    with open(annotations_path, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    return annotations

def analyze_modalities(annotations):
    """Analyze modality distribution in the dataset"""
    modality_counts = {
        'T+A+V': 0,
        'T+A': 0,
        'T+V': 0,
        'T': 0
    }

    for sample in annotations:
        has_audio = len(sample.get('audios', [])) > 0
        has_video = len(sample.get('videos', [])) > 0

        if has_audio and has_video:
            modality_counts['T+A+V'] += 1
        elif has_audio:
            modality_counts['T+A'] += 1
        elif has_video:
            modality_counts['T+A+V'] += 1
        else:
            modality_counts['T'] += 1

    # Remove categories with 0 counts for cleaner visualization
    modality_counts = {k: v for k, v in modality_counts.items() if v > 0}

    return modality_counts

def create_modality_bar_plot(modality_counts):
    """Create bar plot for modality distribution"""

    # Sort by count for better visualization
    sorted_items = sorted(modality_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors[0],
            text=values,
            textposition='auto',
            textfont=dict(size=18, color='white', family='Computer Modern')
        )
    ])

    fig.update_layout(
        title={
            'text': 'Distribution of Modalities in Human Behavior Dataset',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 30, 'color': 'black', 'family': 'Computer Modern'}
        },
        xaxis_title='Modality Combination',
        yaxis_title='Number of Samples',
        template='plotly_white',
        width=800,
        height=800,
        showlegend=False,
        font=dict(size=18, color='black', family='Computer Modern'),
        xaxis={'tickangle': -45},
        margin=dict(l=100, r=100, t=120, b=120)
    )

    # Save the figure with scale=2 for high resolution
    fig.write_html('visualizations/modality_distribution.html')
    fig.write_image('visualizations/modality_distribution.png', width=800, height=800, scale=2)
    print("Modality distribution plot saved to visualizations/modality_distribution.html and .png")

    return fig

def load_metadata_cache():
    """Load existing metadata cache if it exists"""
    if metadata_cache_path.exists():
        try:
            with open(metadata_cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_metadata_cache(metadata_cache):
    """Save metadata cache to file"""
    try:
        with open(metadata_cache_path, 'w') as f:
            json.dump(metadata_cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save metadata cache: {e}")

def get_media_duration(file_path, metadata_cache):
    """Get actual duration of audio/video file using ffprobe, with caching"""

    # Check if duration is already in cache
    if file_path in metadata_cache:
        return metadata_cache[file_path].get('duration')

    # Construct full path
    full_path = Path("/Users/dvd/Downloads/human_behaviour_data") / file_path

    if not full_path.exists():
        # Cache the failure
        metadata_cache[file_path] = {'duration': None, 'exists': False}
        return None

    try:
        # Use ffprobe to get duration
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(full_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())

        # Cache the successful result
        metadata_cache[file_path] = {
            'duration': duration,
            'exists': True,
            'file_size': full_path.stat().st_size
        }

        return duration
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        # Cache the failure
        metadata_cache[file_path] = {'duration': None, 'exists': True, 'error': True}
        return None

def analyze_durations(annotations):
    """Analyze duration distribution of audio/video samples"""
    durations = []
    failed_count = 0

    # Load existing metadata cache
    metadata_cache = load_metadata_cache()
    initial_cache_size = len(metadata_cache)
    print(f"Loaded metadata cache with {initial_cache_size} entries")

    print("Extracting media durations...")

    # Filter samples with audio or video
    samples_with_media = [s for s in annotations if s.get('audios', []) or s.get('videos', [])]

    # Use tqdm for progress bar
    for i, sample in enumerate(tqdm(samples_with_media, desc="Processing media files")):
        # Check if sample has audio or video
        media_files = sample.get('audios', []) + sample.get('videos', [])

        if media_files:
            # Get duration for the first media file
            duration = get_media_duration(media_files[0], metadata_cache)

            if duration is not None:
                durations.append(duration)
            else:
                failed_count += 1

        # Save cache periodically (every 1000 samples)
        if (i + 1) % 1000 == 0:
            save_metadata_cache(metadata_cache)
            new_entries = len(metadata_cache) - initial_cache_size
            tqdm.write(f"  Saved cache with {new_entries} new entries")

    # Save final cache
    save_metadata_cache(metadata_cache)
    new_entries = len(metadata_cache) - initial_cache_size
    print(f"Final cache saved with {new_entries} new entries (total: {len(metadata_cache)})")

    if failed_count > 0:
        print(f"  Warning: Failed to get duration for {failed_count} files")

    return durations

def create_duration_pie_chart(durations):
    """Create pie chart for duration distribution"""

    if not durations:
        print("No duration data available")
        return None

    # Define duration bins based on distribution
    durations_array = np.array(durations)

    # Define bins with nice round numbers
    bins = [0, 5, 10, 20, float('inf')]
    labels = ['<5s', '5-10s', '10-20s', '>20s']

    # Count samples in each bin
    bin_counts = []
    for i in range(len(bins) - 1):
        count = np.sum((durations_array >= bins[i]) & (durations_array < bins[i + 1]))
        bin_counts.append(count)

    # Filter out empty bins
    non_zero_indices = [i for i, count in enumerate(bin_counts) if count > 0]
    labels = [labels[i] for i in non_zero_indices]
    bin_counts = [bin_counts[i] for i in non_zero_indices]

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=bin_counts,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        textposition='auto',
        textfont=dict(size=24, color='white', family='Computer Modern'),
        hovertemplate='<b>%{label}</b><br>' +
                      'Count: %{value}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>'
    )])

    fig.update_layout(
        title={
            'text': 'Distribution of Audio/Video Duration',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 30, 'color': 'black', 'family': 'Computer Modern'}
        },
        template='plotly_white',
        width=800,
        height=800,
        font=dict(size=24, color='white', family='Computer Modern'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=24, color='black', family='Computer Modern')
        ),
        margin=dict(l=100, r=150, t=120, b=100)
    )

    # Save the figure with scale=2 for high resolution
    fig.write_html('visualizations/duration_distribution.html')
    fig.write_image('visualizations/duration_distribution.png', width=800, height=800, scale=2)
    print("Duration distribution pie chart saved to visualizations/duration_distribution.html and .png")

    return fig

def main():
    """Main function to run all analyses"""
    print("Loading annotations...")
    annotations = load_annotations()
    print(f"Loaded {len(annotations)} samples")

    # Figure 1: Modality distribution bar plot
    print("\nAnalyzing modality distribution...")
    modality_counts = analyze_modalities(annotations)
    print("Modality counts:", modality_counts)
    fig1 = create_modality_bar_plot(modality_counts)

    # Figure 2: Duration distribution pie chart
    print("\nAnalyzing duration distribution...")
    durations = analyze_durations(annotations)
    print(f"Analyzed {len(durations)} samples with audio/video")
    fig2 = None
    if durations:
        print(f"Duration statistics: min={min(durations):.2f}s, max={max(durations):.2f}s, mean={np.mean(durations):.2f}s")
        fig2 = create_duration_pie_chart(durations)

    print("\nAll visualizations created successfully!")

    # Show the figures in browser (optional)
    if fig1:
        fig1.show()
    if fig2:
        fig2.show()

if __name__ == "__main__":
    main()