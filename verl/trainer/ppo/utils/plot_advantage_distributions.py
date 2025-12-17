"""
Standalone script to plot TARPO advantage distributions per task.

This script can be used to:
1. Plot advantage distributions from saved JSON files (standalone mode)
2. Generate plots during training (when called from ray_trainer.py)

Usage (standalone):
    python plot_advantage_distributions.py --json_path path/to/step_100_advantages_final.json --output_path output.pdf

Usage (from training code):
    from visuals.plot_advantage_distributions import plot_advantage_distributions
    fig = plot_advantage_distributions(json_data_or_path, save_path=None)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import plotly.graph_objects as go
import plotly.io as pio

# Configure plotly renderer
pio.renderers.default = 'browser'
pio.kaleido.scope.mathjax = None  # Disable MathJax for Kaleido


def load_advantage_data(json_path_or_data: Union[str, Path, Dict]) -> Dict[str, Any]:
    """
    Load advantage data from JSON file or dict.

    Args:
        json_path_or_data: Path to JSON file or dict with advantage data

    Returns:
        Dictionary with advantage data
    """
    if isinstance(json_path_or_data, (str, Path)):
        with open(json_path_or_data, 'r') as f:
            return json.load(f)
    else:
        return json_path_or_data


def plot_advantage_distributions(
    json_path_or_data: Union[str, Path, Dict],
    *,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 600,
) -> go.Figure:
    """
    Create box and whisker plot with individual points for advantage distributions per task.

    Args:
        json_path_or_data: Path to JSON file or dict with advantage data
        save_path: Optional path to save the plot (PDF or image)
        title: Optional custom title for the plot
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        plotly Figure object
    """
    # Load data
    data = load_advantage_data(json_path_or_data)

    step = data.get("step", "unknown")
    advantage_type = data.get("advantage_type", "unknown")
    tasks_data = data.get("tasks", {})

    # Sort tasks alphabetically for consistent ordering
    sorted_tasks = sorted(tasks_data.keys())

    # Prepare traces
    box_traces = []
    scatter_traces = []

    for task_name in sorted_tasks:
        task_info = tasks_data[task_name]
        advantages = task_info["advantages"]

        # Add box plot for the task (without showing individual points in the box)
        box_traces.append(go.Box(
            y=advantages,
            x=[task_name] * len(advantages),
            name=task_name,
            boxpoints=False,  # We'll add points separately with scatter
            marker=dict(color="lightgray"),
            line=dict(color="black"),
            showlegend=False
        ))

        # Add scatter plot for individual points
        scatter_traces.append(go.Scatter(
            y=advantages,
            x=[task_name] * len(advantages),
            mode="markers",
            marker=dict(
                color="steelblue",
                size=6,
                symbol="circle",
                opacity=0.6,
                line=dict(color="darkblue", width=0.5)
            ),
            name=task_name,
            showlegend=False
        ))

    # Create figure
    fig = go.Figure()

    # Add box plots first, then scatter points on top
    fig.add_traces(box_traces)
    fig.add_traces(scatter_traces)

    # Set title
    if title is None:
        title = f"Advantage Distributions per Task (Step {step}, {advantage_type})"

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Task",
            tickfont=dict(size=12),
            tickangle=-45 if len(sorted_tasks) > 10 else 0,
        ),
        yaxis=dict(
            title="Advantage Value",
            tickfont=dict(size=12),
        ),
        width=width,
        height=height,
        margin=dict(t=80, b=120, l=80, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
    )

    # Add gridlines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_xaxes(showgrid=False)

    # Save if path provided
    if save_path:
        # Determine format from extension
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix.lower() == '.pdf':
            fig.write_image(str(save_path), format='pdf')
        elif save_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg']:
            fig.write_image(str(save_path))
        else:
            # Default to PDF
            fig.write_image(str(save_path), format='pdf')

        print(f"Saved plot to {save_path}")

    return fig


def main():
    """Command-line interface for standalone usage."""
    parser = argparse.ArgumentParser(
        description="Plot TARPO advantage distributions from JSON files"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to JSON file with advantage data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output plot (PDF, PNG, etc.). If not provided, displays interactively."
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="Figure width in pixels (default: 1200)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Figure height in pixels (default: 600)"
    )

    # For comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable comparison mode for multiple advantage types"
    )
    parser.add_argument(
        "--json_paths",
        nargs="+",
        default=None,
        help="Multiple JSON paths for comparison (use with --compare)"
    )
    parser.add_argument(
        "--advantage_types",
        nargs="+",
        default=None,
        help="Labels for advantage types in comparison (use with --compare)"
    )

    args = parser.parse_args()

    fig = plot_advantage_distributions(
            args.json_path,
            save_path=args.output_path,
            title=args.title,
            width=args.width,
            height=args.height
        )

    # Show interactively if no output path
    if args.output_path is None:
        fig.show()


if __name__ == "__main__":
    main()
