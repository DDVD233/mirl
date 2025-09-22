import json
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
import numpy as np
from collections import defaultdict

# Define color palette for modalities
modality_colors = {
    'T/A/V': '#7dbfa7',  # Text + Audio + Video
    'A': '#da81c1',      # Audio only
    'T': '#b0d766'       # Text only
}

# Define task to dataset mapping based on the table
task_dataset_mapping = {
    'SEN': ['CH-SIMSv2', 'MELD-S', 'MOSEI-S'],
    'EMO': ['CREMA-D', 'RAVDESS', 'TESS', 'MELD-E', 'MOSEI-E'],
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
    'MOSEI-E': 'T/A/V',
    'MOSEI-S': 'T/A/V',
    'MELD-E': 'T/A/V',
    'MELD-S': 'T/A/V',
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
                'mosei_emotion': 'MOSEI-E',
                'mosei_senti': 'MOSEI-S',
                'meld_emotion': 'MELD-E',
                'meld_senti': 'MELD-S',
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
            elif 'senti' in original_dataset:
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
    """Assign color based on modality type"""
    # Map all modality combinations to the three main categories
    if modality == 'T':
        return modality_colors['T']  # Text only
    elif modality == 'A':
        return modality_colors['A']  # Audio only
    else:
        # All other combinations (T/A, A/V, T/V, T/A/V) use the T/A/V color
        return modality_colors['T/A/V']

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

    # Track which legend groups we've already added
    legend_added = {'T': False, 'A': False, 'T/A/V': False}

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

            # Determine legend group and whether to show in legend
            if modality == 'T':
                legend_group = 'Text only'
                show_legend = not legend_added['T']
                legend_added['T'] = True
            elif modality == 'A':
                legend_group = 'Audio only'
                show_legend = not legend_added['A']
                legend_added['A'] = True
            else:
                legend_group = 'T+A+V'
                show_legend = not legend_added['T/A/V']
                legend_added['T/A/V'] = True

            # Create trace with mixed text positions
            trace = go.Bar(
                name=legend_group if show_legend else f"{dataset} ({modality})",
                x=tasks,
                y=task_values,
                marker_color=color,
                text=text_labels,
                textposition=text_positions,
                textfont=dict(size=31, family='Computer Modern'),
                customdata=[f"{dataset} ({modality})"] * len(tasks),
                legendgroup=legend_group,
                showlegend=show_legend
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
        # title={
        #     'text': 'Dataset Contributions Across Behavioral Tasks',
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'font': {'size': 36, 'color': 'black', 'family': 'Computer Modern'}
        # },
        xaxis_title='Task',
        yaxis_title='Number of Samples',
        barmode='stack',
        template='plotly_white',
        width=800,
        height=800,
        font=dict(size=24, color='black', family='Computer Modern'),
        showlegend=True,  # Show legend
        legend=dict(
            x=1,  # Position at right
            y=0.9,  # Position at top
            xanchor='right',
            yanchor='top',
            font=dict(size=24, family='Computer Modern'),
            bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
            bordercolor='black',
            borderwidth=1
        ),
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
    fig.write_image('visualizations/task_dataset_distribution.png', width=700, height=900, scale=4)
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