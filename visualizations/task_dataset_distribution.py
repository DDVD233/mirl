import json
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np
from collections import defaultdict

# Define color palette
colors = ["#da81c1", "#7dbfa7", "#b0d766", "#8ca0cb", "#ee946c", "#da81c1"]

# Define task to dataset mapping based on the table
task_dataset_mapping = {
    'SEN': ['MOSEI', 'MELD', 'CH-SIMSv2'],
    'EMO': ['MOSEI', 'MELD', 'CREMA-D', 'RAVDESS', 'TESS'],
    'HUM': ['UR-FUNNY'],
    'SAR': ['MUSTARD'],
    'SOC': ['SocialIQ2'],
    'INT': ['IntentQA'],
    'NVC': ['MimeQA'],
    'ANX': ['MMPsy'],
    'DEP': ['DAIC-WOZ', 'MMPsy'],
    'PTSD': ['PTSDW']
}

# Define dataset to modality mapping (T=Text, A=Audio, V=Video)
dataset_modalities = {
    'MOSEI': 'T/A/V',
    'MELD': 'T/A/V',
    'CH-SIMSv2': 'T/A/V',
    'CREMA-D': 'A',
    'RAVDESS': 'A/V',
    'TESS': 'A',
    'UR-FUNNY': 'T/A/V',
    'MUSTARD': 'T/A/V',
    'SocialIQ2': 'T/A/V',
    'IntentQA': 'T/A/V',
    'MimeQA': 'T/A/V',
    'MMPsy': 'T',
    'DAIC-WOZ': 'T/A',
    'PTSDW': 'T/A/V'
}

# Load annotations to get actual sample counts per dataset and task
annotations_path = Path("/Users/dvd/Downloads/human_behaviour_data/all_annotations.jsonl")

def get_dataset_task_counts():
    """Count samples per dataset and task from annotations"""
    dataset_task_counts = defaultdict(lambda: defaultdict(int))

    print("Loading annotations to count samples per dataset and task...")
    with open(annotations_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            dataset = sample.get('dataset', '')

            # Map dataset names to our standardized names
            dataset_mapping = {
                'mosei_emotion': 'MOSEI',
                'mosei_sentiment': 'MOSEI',
                'meld_emotion': 'MELD',
                'meld_sentiment': 'MELD',
                'cremad': 'CREMA-D',
                'tess': 'TESS',
                'chsimsv2': 'CH-SIMSv2',
                'urfunny': 'UR-FUNNY',
                'mmsd': 'MUSTARD',
                'siq2': 'SocialIQ2',
                'intentqa': 'IntentQA',
                'mimeqa': 'MimeQA',
                'mmpsy_anxiety': 'MMPsy',
                'mmpsy_depression': 'MMPsy',
                'daicwoz_with_transcript': 'DAIC-WOZ',
                'ptsd_in_the_wild': 'PTSDW'
            }

            # Get standardized dataset name
            for key, value in dataset_mapping.items():
                if key in dataset.lower():
                    dataset = value
                    break

            # Determine task based on dataset and original dataset name
            original_dataset = sample.get('dataset', '').lower()
            if 'emotion' in original_dataset:
                task = 'EMO'
            elif 'sentiment' in original_dataset:
                task = 'SEN'
            elif 'anxiety' in original_dataset:
                task = 'ANX'
            elif 'depression' in original_dataset:
                task = 'DEP'
            else:
                # Map based on dataset to task
                for t, datasets in task_dataset_mapping.items():
                    if dataset in datasets:
                        task = t
                        break

            if dataset and task:
                dataset_task_counts[task][dataset] += 1

    return dataset_task_counts

def assign_modality_colors(modality):
    """Assign color based on modality complexity"""
    modality_color_map = {
        'T': colors[0],       # Text only - lightest
        'A': colors[1],       # Audio only
        'V': colors[2],       # Video only
        'T/A': colors[3],     # Text + Audio
        'A/V': colors[4],     # Audio + Video
        'T/V': colors[5],     # Text + Video
        'T/A/V': colors[2]    # All three - darkest
    }
    return modality_color_map.get(modality, colors[0])

def create_stacked_bar_chart():
    """Create stacked bar chart showing dataset contributions to each task"""

    # Get actual counts from annotations
    dataset_task_counts = get_dataset_task_counts()

    # Calculate total samples per task for ranking
    task_totals = {}
    for task in ['SEN', 'EMO', 'HUM', 'SAR', 'SOC', 'INT', 'NVC', 'ANX', 'DEP', 'PTSD']:
        task_totals[task] = sum(dataset_task_counts[task].values())

    # Sort tasks by total samples (largest to smallest)
    tasks = sorted(task_totals.keys(), key=lambda x: task_totals[x], reverse=True)

    # Group datasets by modality for consistent ordering
    modality_groups = {
        'T': [],
        'A': [],
        'V': [],
        'T/A': [],
        'A/V': [],
        'T/V': [],
        'T/A/V': []
    }

    all_datasets = set()
    for datasets in task_dataset_mapping.values():
        all_datasets.update(datasets)

    # Organize datasets by modality
    for dataset in all_datasets:
        modality = dataset_modalities.get(dataset, 'T')
        if modality in modality_groups:
            modality_groups[modality].append(dataset)

    # Sort datasets within each modality group
    for modality in modality_groups:
        modality_groups[modality].sort()

    traces = []

    # Create traces in modality order for consistent stacking
    modality_order = ['T', 'A', 'V', 'T/A', 'A/V', 'T/V', 'T/A/V']

    for modality in modality_order:
        for dataset in modality_groups[modality]:
            task_values = []
            text_labels = []
            text_positions = []
            text_colors = []

            for task in tasks:
                if dataset in task_dataset_mapping[task]:
                    count = dataset_task_counts[task].get(dataset, 0)
                    task_values.append(count)
                    if count > 0:
                        # Create multiline label if needed
                        if len(dataset) > 10:
                            # Split long names
                            parts = dataset.split('-')
                            if len(parts) > 1:
                                label = '<br>'.join(parts)
                            else:
                                label = dataset[:10] + '<br>' + dataset[10:]
                        else:
                            label = dataset

                        # Determine text position and color based on count
                        if count < 2700:  # Small segments - text above in black
                            text_labels.append(f"<b>{label}</b><br>{count:,}")
                            text_positions.append('outside')
                            text_colors.append('black')
                        else:  # Large segments - text inside in white
                            text_labels.append(f"<b>{label}</b><br>{count:,}")
                            text_positions.append('inside')
                            text_colors.append('white')
                    else:
                        text_labels.append("")
                        text_positions.append('inside')
                        text_colors.append('white')
                else:
                    task_values.append(0)
                    text_labels.append("")
                    text_positions.append('inside')
                    text_colors.append('white')

            # Get color for modality
            color = assign_modality_colors(modality)

            # Create trace with mixed text positions
            trace = go.Bar(
                name=f"{dataset} ({modality})",
                x=tasks,
                y=task_values,
                marker_color=color,
                text=text_labels,
                textposition=text_positions,
                textfont=dict(size=24, family='Computer Modern'),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Task: %{x}<br>' +
                              'Samples: %{y:,}<br>' +
                              '<extra></extra>'
            )

            # Update text colors individually
            for i, tc in enumerate(text_colors):
                if i < len(trace.text):
                    trace.textfont = dict(size=24, family='Computer Modern', color=text_colors)

            traces.append(trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title={
            'text': 'Dataset Contributions Across Behavioral Tasks',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 36, 'color': 'black', 'family': 'Computer Modern'}
        },
        xaxis_title='Task',
        yaxis_title='Number of Samples',
        barmode='stack',
        template='plotly_white',
        width=800,
        height=800,
        font=dict(size=24, color='black', family='Computer Modern'),
        showlegend=False,  # Remove legend
        bargap=0.1,  # Smaller bar spacing
        margin=dict(l=120, r=100, t=150, b=120),
        xaxis=dict(
            tickfont=dict(size=24, family='Computer Modern'),
            title_font=dict(size=28, family='Computer Modern')
        ),
        yaxis=dict(
            tickfont=dict(size=24, family='Computer Modern'),
            title_font=dict(size=28, family='Computer Modern'),
            gridcolor='lightgray',
            gridwidth=0.5
        )
    )

    # Add task descriptions as annotations below x-axis
    task_descriptions = {
        'SEN': 'Sentiment',
        'EMO': 'Emotion',
        'HUM': 'Humor',
        'SAR': 'Sarcasm',
        'SOC': 'Social',
        'INT': 'Intent',
        'NVC': 'Non-Verbal',
        'ANX': 'Anxiety',
        'DEP': 'Depression',
        'PTSD': 'PTSD'
    }

    # for i, task in enumerate(tasks):
    #     fig.add_annotation(
    #         x=i,
    #         y=-0.12,
    #         xref='x',
    #         yref='paper',
    #         # text=task_descriptions[task],
    #         showarrow=False,
    #         font=dict(size=16, color='gray', family='Computer Modern'),
    #         xanchor='center'
    #     )

    # Save the figure
    fig.write_html('visualizations/task_dataset_distribution.html')
    fig.write_image('visualizations/task_dataset_distribution.png', width=800, height=800, scale=2)
    print("Task-dataset distribution chart saved!")

    # Print summary statistics (ordered by size)
    print("\nSummary Statistics (ordered by size):")
    for task in tasks:
        total = sum(dataset_task_counts[task].values())
        if total > 0:
            print(f"{task}: {total:,} samples across {len(dataset_task_counts[task])} datasets")

    return fig

def main():
    print("Creating task-dataset distribution visualization...")
    fig = create_stacked_bar_chart()
    fig.show()

if __name__ == "__main__":
    main()