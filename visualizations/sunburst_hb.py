import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------- Data Definitions -------------------------
# Define the datasets with their properties
datasets_info = {
    "CMU-MOSEI": {
        "modalities": "TAV",
        "tasks": ["EMO", "SEN"],
        "samples": {"EMO": 8598, "SEN": 22856}
    },
    "MELD": {
        "modalities": "TAV",
        "tasks": ["EMO", "SEN"],
        "samples": {"EMO": 13706, "SEN": 13705}
    },
    "TESS": {
        "modalities": "TA",
        "tasks": ["EMO"],
        "samples": 2800
    },
    "CREMA-D": {
        "modalities": "TA",
        "tasks": ["EMO"],
        "samples": 7442
    },
    "CH-SIMSv2": {
        "modalities": "TAV",
        "tasks": ["SEN"],
        "samples": 4403
    },
    "Social-IQ 2.0": {
        "modalities": "TAV",
        "tasks": ["SOC"],
        "samples": 6437
    },
    "IntentQA": {
        "modalities": "TAV",
        "tasks": ["INT"],
        "samples": 16297
    },
    "MimeQA": {
        "modalities": "TAV",
        "tasks": ["NVC"],
        "samples": 806
    },
    "UR-FUNNYv2": {
        "modalities": "TAV",
        "tasks": ["HUM"],
        "samples": 2125
    },
    "MUStARD": {
        "modalities": "TAV",
        "tasks": ["SAR"],
        "samples": 690
    },
    "DAIC-WOZ": {
        "modalities": "TA",
        "tasks": ["DEP"],
        "samples": 189
    },
    "MMPsy": {
        "modalities": "T",
        "tasks": ["DEP", "ANX"],
        "samples": {"DEP": 1275, "ANX": 1275}
    },
    "PTSDIW": {
        "modalities": "TAV",
        "tasks": ["PTSD"],
        "samples": 634
    }
}

# Define colors for modalities
modality_colors = {
    "TAV": "#7dbfa7",  # Teal/green
    "TA": "#b0d766",  # Light green
    "T": "#8ca0cb"  # Blue
}

# No need for separate task colors - will use modality colors for bars

# ------------------------- STEP 1: BUILD DATAFRAME -------------------------
rows = []
dataset_order = []

# Process each dataset
for dataset_name, info in datasets_info.items():
    modalities = info["modalities"]
    tasks = info["tasks"]

    # Handle datasets with multiple tasks
    if len(tasks) > 1:
        task_suffixes = {"EMO": "-E", "SEN": "-S", "DEP": "-D", "ANX": "-A"}
        for task in tasks:
            suffix = task_suffixes.get(task, f"-{task}")
            full_name = f"{dataset_name.replace('CMU-', '')}{suffix}"

            if isinstance(info["samples"], dict):
                samples = info["samples"][task]
            else:
                samples = info["samples"]

            dataset_order.append((modalities, task, full_name))
            rows.append({
                "Modality": modalities,
                "Task": task,
                "Dataset": full_name,
                "Samples": samples,
                "LeafCount": 1
            })
    else:
        task = tasks[0]
        dataset_order.append((modalities, task, dataset_name))
        rows.append({
            "Modality": modalities,
            "Task": task,
            "Dataset": dataset_name,
            "Samples": info["samples"] if isinstance(info["samples"], int) else info["samples"][task],
            "LeafCount": 1
        })

df = pd.DataFrame(rows)

# Sort by modality order (TAV, TA, T), then by samples descending
modality_order = ["TAV", "TA", "T"]
df["modality_sort"] = df["Modality"].map({m: i for i, m in enumerate(modality_order)})
df = df.sort_values(["modality_sort", "Samples"], ascending=[True, False])
df = df.drop("modality_sort", axis=1)

# Update dataset_order based on sorted dataframe
dataset_order = [(row["Modality"], row["Task"], row["Dataset"]) for _, row in df.iterrows()]

# Scale LeafCount values for proper ordering
total_datasets = len(dataset_order)
for idx in range(len(df)):
    df.loc[df.index[idx], "LeafCount"] = 1 - (idx / (total_datasets * 10000))

# ------------------------- STEP 2: CREATE SUNBURST -------------------------
fig_sb = px.sunburst(
    df,
    path=["Modality", "Task", "Dataset"],
    values="LeafCount",
    color="Modality",
    color_discrete_map=modality_colors,
    custom_data=["Samples"],
)

fig = go.Figure(data=fig_sb.data, layout=fig_sb.layout)
sizes = [0.14, 0.86]
fig.update_traces(
    domain=dict(x=sizes, y=sizes),
    textfont=dict(color='rgba(0, 0, 0, 0.7)', size=35),
    insidetextorientation='radial'
)


# ------------------------- STEP 3: ADD BAR PLOT -------------------------
def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def get_log2_height(val, min_val, max_val, base_height=1.0, scale=0.5):
    if val <= 0 or max_val <= 0 or min_val == max_val:
        return base_height
    # Use min_val/2 as the base to ensure even the smallest value has some bar height
    adjusted_min = min_val / 2
    return base_height + scale * (np.log2(val) - np.log2(adjusted_min)) / (np.log2(max_val) - np.log2(adjusted_min))


# Prepare data for bar plot
all_datasets = df[["Modality", "Task", "Dataset", "Samples"]].drop_duplicates()
all_datasets["order_idx"] = all_datasets.apply(
    lambda row: dataset_order.index((row["Modality"], row["Task"], row["Dataset"])),
    axis=1
)
all_datasets = all_datasets.sort_values("order_idx")

n_ds = len(all_datasets)
angle_step = 360 / n_ds
sector_width = angle_step * 0.95

min_samps = all_datasets["Samples"].replace(0, np.nan).min() or 1
max_samps = all_datasets["Samples"].max() or 1

start_angle = 11.5
current_angle = start_angle

for _, row in all_datasets.iterrows():
    ds_name = row["Dataset"]
    ds_samps = row["Samples"]
    ds_modality = row["Modality"]
    ds_task = row["Task"]

    # Use modality color for the bars (matching the sunburst)
    color_hex = modality_colors.get(ds_modality, "#888888")
    color_rgb = hex_to_rgb(color_hex)

    center_angle = current_angle
    current_angle += angle_step

    base_height = 1.16
    r_outer_val = get_log2_height(ds_samps, min_samps, max_samps, base_height=base_height, scale=0.5)

    theta = np.linspace(center_angle - sector_width / 2, center_angle + sector_width / 2, 20)
    r_inner = np.ones_like(theta) + base_height - 1
    r_outer = np.full_like(theta, r_outer_val)

    fig.add_trace(
        go.Scatterpolar(
            r=np.concatenate([r_inner, r_outer[::-1]]),
            theta=np.concatenate([theta, theta[::-1]]),
            mode="lines",
            fill="toself",
            fillcolor=f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.7)",
            line=dict(color=f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},0.7)", width=0),
            hovertemplate=(
                    "<b>Dataset:</b> %{text}<br>"
                    "<b>Samples:</b> " + str(ds_samps) + "<br>"
                                                         "<b>Modality:</b> " + ds_modality + "<br>"
                                                                                             "<b>Task:</b> " + ds_task + "<extra></extra>"
            ),
            text=ds_name,
            showlegend=False
        )
    )

# ------------------------- STEP 4: FINAL LAYOUT -------------------------
fig.update_layout(
    width=2000,
    height=2000,
    # title="Multimodal Dataset Distribution by Modality, Task, and Dataset",
    font=dict(size=40, color="black"),
    polar=dict(
        radialaxis=dict(range=[0, 1.6], visible=False),
        angularaxis=dict(visible=False),
        bgcolor="rgba(0,0,0,0)"
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=50, r=50, t=100, b=50)
)

# Uncomment the following line to save the figure
fig.write_image("sunburst_multimodal_distribution.png")

# Display the figure
fig.show()