import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import time
import plotly.express as px

fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("random.pdf")
time.sleep(1)

# Data from FairGRPO (Paste 1)
fairgrpo_data = {'fairness/ham10000/gender/f1': 0.28796398771297904, 'fairness/chexpert_full/age/a1/f1': 0.41809523809523813, 'fairness/chexpert_full/age/a3/f1': 0.37717422890817187, 'fairness/isic2020/age/a2/f1': 0.49576558265582654, 'fairness/vindr/age/a2/accuracy': 0.8286061588330632, 'fairness/vindr/age/a1/accuracy': 0.8018518518518517, 'fairness/isic2020/age/a3/f1': 0.5734675826639041, 'fairness/pad_ufes_20/gender/male/f1': 0.21964138051094576, 'fairness/isic2020/age/f1': 0.5396401973426159, 'fairness/COVID-BLUES/age/f1': 0, 'fairness/pad_ufes_20/gender/female/accuracy': 0.8046153846153846, 'fairness/ham10000/age/a3/f1': 0.30878936497746406, 'fairness/isic2020/gender/female/f1': 0.5524991948827861, 'fairness/hemorrhage/age/a3/accuracy': 0.8314606741573034, 'fairness/ham10000/avg_f1': 0.26808741766595645, 'fairness/hemorrhage/gender/male/f1': 0.6720350166347602, 'fairness/hemorrhage/age/a2/f1': 0.6289760348583877, 'fairness/hemorrhage/gender/female/f1': 0.8, 'fairness/hemorrhage/gender/male/accuracy': 0.7776381909547738, 'fairness/vindr/age/a1/f1': 0.22823355506282333, 'fairness/vindr/age/a3/f1': 0.21738962998995928, 'fairness/ham10000/age/a4/accuracy': 0.7224137931034482, 'fairness/vindr/age/a3/accuracy': 0.8085774058577405, 'fairness/COVID-BLUES/age/a4/f1': 0, 'fairness/chexpert_full/gender/male/f1': 0.3396422298799233, 'fairness/chexpert_full/gender/female/f1': 0.39780167173357267, 'fairness/ham10000/age/a2/accuracy': 0.8976995940460082, 'fairness/hemorrhage/age/a3/f1': 0.8064375815571987, 'fairness/hemorrhage/age/a2/accuracy': 0.7981818181818182, 'fairness/COVID-BLUES/age/a3/f1': 0, 'fairness/ham10000/gender/male/f1': 0.2633335796039219, 'fairness/pad_ufes_20/age/a2/f1': 0.3912442396313365, 'fairness/hemorrhage/avg_f1': 0.723267293170722, 'fairness/COVID-BLUES/age/a3/accuracy': 0.5, 'fairness/ham10000/gender/female/accuracy': 0.8710765239948121, 'fairness/hemorrhage/gender/female/accuracy': 0.8888888888888888, 'fairness/COVID-BLUES/age/a2/accuracy': 0.5, 'fairness/hemorrhage/age/f1': 0.7105170780240638, 'fairness/isic2020/age/a3/accuracy': 0.9710004833252779, 'fairness/ham10000/gender/female/f1': 0.3125943958220362, 'fairness/chexpert_full/gender/female/accuracy': 0.7838765008576328, 'fairness/pad_ufes_20/gender/male/accuracy': 0.8100000000000002, 'fairness/chexpert_full/age/a2/accuracy': 0.8173758865248226, 'fairness/pad_ufes_20/age/a3/accuracy': 0.7638766519823789, 'fairness/vindr/age/f1': 0.22429785336575508, 'fairness/chexpert_full/age/f1': 0.3825917851435373, 'fairness/isic2020/age/a1/accuracy': 0.9815789473684211, 'fairness/pad_ufes_20/age/a4/accuracy': 0.7572463768115942, 'fairness/chexpert_full/age/a4/accuracy': 0.7145454545454545, 'fairness/isic2020/age/a1/f1': 0.4940079893475366, 'fairness/vindr/age/a4/f1': 0.2051282051282051, 'fairness/pad_ufes_20/gender/f1': 0.23856961309135222, 'fairness/isic2020/gender/male/accuracy': 0.9776145505421476, 'fairness/COVID-BLUES/avg_f1': 0, 'fairness/pad_ufes_20/age/a3/f1': 0.19682157381450624, 'fairness/hemorrhage/age/a1/f1': 0.696137617656605, 'fairness/pad_ufes_20/gender/female/f1': 0.2574978456717587, 'fairness/COVID-BLUES/age/a4/accuracy': 0.5, 'fairness/isic2020/avg_f1': 0.5474326947316872, 'fairness/ham10000/age/a2/f1': 0.18593644354293443, 'fairness/chexpert_full/age/a4/f1': 0.36757765534137604, 'fairness/isic2020/gender/f1': 0.5552251921207585, 'fairness/COVID-BLUES/age/a2/f1': 0, 'fairness/isic2020/gender/female/accuracy': 0.9791433891992551, 'fairness/hemorrhage/gender/f1': 0.7360175083173801, 'fairness/ham10000/age/a4/f1': 0.1957120836735503, 'fairness/ham10000/age/a1/f1': 0.3024054982817869, 'fairness/vindr/age/a4/accuracy': 0.5555555555555557, 'fairness/pad_ufes_20/age/f1': 0.32094435039982283, 'fairness/pad_ufes_20/age/a2/accuracy': 0.7884615384615384, 'fairness/isic2020/gender/male/f1': 0.5579511893587308, 'fairness/vindr/avg_f1': 0.22429785336575508, 'fairness/chexpert_full/gender/f1': 0.36872195080674797, 'fairness/chexpert_full/gender/male/accuracy': 0.8235294117647058, 'fairness/ham10000/age/a3/accuracy': 0.8148905109489052, 'fairness/chexpert_full/avg_f1': 0.37565686797514264, 'fairness/pad_ufes_20/age/a1/f1': 0.4615384615384615, 'fairness/hemorrhage/age/a1/accuracy': 0.7947019867549668, 'fairness/ham10000/age/f1': 0.24821084761893392, 'fairness/isic2020/age/a4/f1': 0.5953196347031964, 'fairness/chexpert_full/age/a3/accuracy': 0.8185840707964601, 'fairness/pad_ufes_20/avg_f1': 0.2797569817455875, 'fairness/pad_ufes_20/age/a4/f1': 0.23417312661498707, 'fairness/ham10000/gender/male/accuracy': 0.8378132118451024, 'fairness/ham10000/age/a1/accuracy': 0.8939393939393939, 'fairness/isic2020/age/a4/accuracy': 0.8940677966101696, 'fairness/vindr/age/a2/f1': 0.2464400232820326, 'fairness/chexpert_full/age/a1/accuracy': 0.8866666666666667, 'fairness/isic2020/age/a2/accuracy': 0.9863957003694994, 'fairness/pad_ufes_20/age/a1/accuracy': 0.875, 'fairness/chexpert_full/age/a2/f1': 0.36752001822936314}

# Data from GRPO (Paste 2)
grpo_data = {'fairness/chexpert_full/age/a1/f1': 0.3180952380952381, 'fairness/chexpert_full/age/a3/f1': 0.2832149012529472, 'fairness/ham10000/age/f1': 0.2626347053444567, 'fairness/pad_ufes_20/age/a3/accuracy': 0.7506607929515419, 'fairness/isic2020/gender/female/accuracy': 0.9813780260707634, 'fairness/ham10000/gender/male/accuracy': 0.8535307517084283, 'fairness/pad_ufes_20/age/a2/f1': 0.38523391812865493, 'fairness/hemorrhage/avg_f1': 0.6834283098143914, 'fairness/COVID-BLUES/age/a4/f1': 0, 'fairness/vindr/age/a3/accuracy': 0.80836820083682, 'fairness/isic2020/age/a1/accuracy': 0.9828947368421053, 'fairness/pad_ufes_20/age/a4/accuracy': 0.75, 'fairness/chexpert_full/gender/f1': 0.2862231097479927, 'fairness/ham10000/age/a4/f1': 0.1845826932923707, 'fairness/chexpert_full/age/f1': 0.2998586737192515, 'fairness/vindr/age/a1/accuracy': 0.8037037037037038, 'fairness/chexpert_full/avg_f1': 0.29304089173362213, 'fairness/chexpert_full/gender/male/f1': 0.2527961218916144, 'fairness/vindr/age/a1/f1': 0.24333333333333335, 'fairness/isic2020/age/a4/accuracy': 0.885593220338983, 'fairness/chexpert_full/age/a1/accuracy': 0.8933333333333335, 'fairness/isic2020/age/f1': 0.5058809408152001, 'fairness/ham10000/avg_f1': 0.25698437873587476, 'fairness/vindr/age/a4/accuracy': 0.6944444444444443, 'fairness/pad_ufes_20/age/a3/f1': 0.18978334327171537, 'fairness/pad_ufes_20/age/a1/f1': 0.4615384615384615, 'fairness/COVID-BLUES/age/f1': 0, 'fairness/hemorrhage/age/a1/f1': 0.7206736353077816, 'fairness/chexpert_full/age/a2/f1': 0.2958370820212925, 'fairness/isic2020/avg_f1': 0.5185570628653358, 'fairness/pad_ufes_20/age/a2/accuracy': 0.7788461538461539, 'fairness/ham10000/age/a2/accuracy': 0.9431664411366711, 'fairness/hemorrhage/age/f1': 0.6664234700272463, 'fairness/chexpert_full/gender/female/f1': 0.31965009760437096, 'fairness/isic2020/gender/male/accuracy': 0.9788387548093739, 'fairness/isic2020/age/a2/f1': 0.49619224911152476, 'fairness/COVID-BLUES/age/a2/accuracy': 0.5, 'fairness/hemorrhage/gender/female/f1': 0.772636815920398, 'fairness/hemorrhage/age/a3/f1': 0.6790169428411841, 'fairness/chexpert_full/age/a2/accuracy': 0.8138297872340425, 'fairness/hemorrhage/age/a3/accuracy': 0.7865168539325842, 'fairness/ham10000/gender/female/f1': 0.2622238994421499, 'fairness/pad_ufes_20/age/a1/accuracy': 0.875, 'fairness/COVID-BLUES/age/a4/accuracy': 0.5, 'fairness/chexpert_full/gender/male/accuracy': 0.8235294117647058, 'fairness/chexpert_full/age/a3/accuracy': 0.8244837758112095, 'fairness/vindr/age/a3/f1': 0.1953387561036346, 'fairness/pad_ufes_20/gender/female/accuracy': 0.796923076923077, 'fairness/ham10000/age/a3/accuracy': 0.8017518248175184, 'fairness/pad_ufes_20/gender/f1': 0.23051800936914418, 'fairness/chexpert_full/age/a4/f1': 0.3022874735075283, 'fairness/isic2020/age/a4/f1': 0.46846846846846846, 'fairness/vindr/avg_f1': 0.2277365708269517, 'fairness/isic2020/gender/f1': 0.5312331849154716, 'fairness/isic2020/age/a3/accuracy': 0.9741420976317061, 'fairness/isic2020/age/a3/f1': 0.5641821946169773, 'fairness/pad_ufes_20/age/a4/f1': 0.22045237336731444, 'fairness/ham10000/age/a2/f1': 0.2615673015109654, 'fairness/vindr/age/a2/accuracy': 0.8409238249594815, 'fairness/pad_ufes_20/gender/female/f1': 0.2471096934912084, 'fairness/isic2020/age/a2/accuracy': 0.987571380584481, 'fairness/hemorrhage/age/a2/accuracy': 0.8363636363636364, 'fairness/COVID-BLUES/age/a2/f1': 0, 'fairness/hemorrhage/gender/male/f1': 0.6282294832826747, 'fairness/vindr/age/f1': 0.2277365708269517, 'fairness/hemorrhage/age/a1/accuracy': 0.8576158940397351, 'fairness/COVID-BLUES/age/a3/f1': 0, 'fairness/ham10000/gender/male/f1': 0.24044420481243586, 'fairness/COVID-BLUES/avg_f1': 0, 'fairness/hemorrhage/gender/female/accuracy': 0.9017094017094017, 'fairness/vindr/age/a2/f1': 0.23417895577560072, 'fairness/chexpert_full/age/a4/accuracy': 0.6909090909090909, 'fairness/ham10000/age/a4/accuracy': 0.710344827586207, 'fairness/vindr/age/a4/f1': 0.2380952380952381, 'fairness/COVID-BLUES/age/a3/accuracy': 0.5, 'fairness/pad_ufes_20/gender/male/f1': 0.21392632524707994, 'fairness/chexpert_full/gender/female/accuracy': 0.7787307032590051, 'fairness/ham10000/gender/f1': 0.2513340521272929, 'fairness/hemorrhage/age/a2/f1': 0.5995798319327731, 'fairness/pad_ufes_20/avg_f1': 0.2723850167228404, 'fairness/ham10000/age/a1/accuracy': 0.9181818181818183, 'fairness/isic2020/gender/female/f1': 0.5168560011715726, 'fairness/ham10000/gender/female/accuracy': 0.8850843060959793, 'fairness/ham10000/age/a1/f1': 0.3825317855168602, 'fairness/hemorrhage/gender/male/accuracy': 0.8140703517587939, 'fairness/pad_ufes_20/gender/male/accuracy': 0.7933333333333333, 'fairness/ham10000/age/a3/f1': 0.22185704105763046, 'fairness/isic2020/age/a1/f1': 0.49468085106382975, 'fairness/pad_ufes_20/age/f1': 0.3142520240765366, 'fairness/hemorrhage/gender/f1': 0.7004331496015364, 'fairness/isic2020/gender/male/f1': 0.5456103686593705}


# Extract F1 scores only and compute differences
def extract_f1_scores(data):
    """Extract only F1 scores from the data"""
    f1_scores = {}
    for key, value in data.items():
        if key.endswith('/f1'):
            f1_scores[key] = value
    return f1_scores


# Get F1 scores
fairgrpo_f1 = extract_f1_scores(fairgrpo_data)
grpo_f1 = extract_f1_scores(grpo_data)

# Compute percentage differences ((FairGRPO - GRPO) / GRPO * 100)
differences = {}
for key in fairgrpo_f1:
    if key in grpo_f1:
        # Avoid division by zero - skip if GRPO value is 0
        if grpo_f1[key] != 0:
            differences[key] = ((fairgrpo_f1[key] - grpo_f1[key]) / grpo_f1[key]) * 100
        else:
            # If GRPO is 0 but FairGRPO is not, this is infinite improvement
            # We'll skip these for now or could set to a large value
            if fairgrpo_f1[key] != 0:
                differences[key] = 100  # Cap at 100% improvement when baseline is 0
            else:
                differences[key] = 0

# Parse and organize data by dataset
datasets = {}
datasets_original_labels = {}  # Store original labels for color mapping
for key, diff in differences.items():
    parts = key.split('/')
    if len(parts) >= 3:
        dataset = parts[1]
        # Skip COVID-BLUES dataset
        if dataset == 'COVID-BLUES':
            continue
        if 'avg' not in key:  # Skip average metrics
            demographic_type = parts[2]  # 'age' or 'gender'

            # Skip aggregated age and gender metrics (those without specific subgroups)
            if len(parts) == 4 and parts[3] == 'f1':
                # This is an overall demographic f1 (e.g., fairness/vindr/age/f1)
                # Skip these aggregated metrics
                continue
            elif len(parts) == 5 and parts[4] == 'f1':
                # This is a specific group f1 (e.g., fairness/vindr/age/a1/f1)
                group = f'{demographic_type}_{parts[3]}'
            else:
                continue

            if dataset not in datasets:
                datasets[dataset] = {}
                datasets_original_labels[dataset] = {}

            # Store original label for color mapping
            datasets_original_labels[dataset][group] = diff

            # Simplify label for display
            display_label = group.split('_')[-1] if '_' in group else group
            datasets[dataset][display_label] = diff

# Create 2x3 subplots with custom order (swapping pad_ufes_20 and hemorrhage)
rows = 2
cols = 3

# Reorder datasets to swap pad_ufes_20 and hemorrhage
dataset_order = list(datasets.keys())
# Find indices of pad_ufes_20 and hemorrhage
if 'pad_ufes_20' in dataset_order and 'hemorrhage' in dataset_order:
    pad_idx = dataset_order.index('pad_ufes_20')
    hem_idx = dataset_order.index('hemorrhage')
    # Swap them
    dataset_order[pad_idx], dataset_order[hem_idx] = dataset_order[hem_idx], dataset_order[pad_idx]

# Create reordered datasets dictionary
datasets_ordered = {k: datasets[k] for k in dataset_order}
datasets_original_labels_ordered = {k: datasets_original_labels[k] for k in dataset_order}

# Create subplot titles with uppercase formatting
subplot_titles = [dataset.replace('isic', 'ISIC').replace('ham', 'HAM').replace('pad_ufes_20', 'PAD-UFES-20').replace('vindr',
                                                                                                                'VinDr-Mammo').replace(
                      'chexpert_full', 'CheXpert').replace('hemorrhage', 'HEMORRHAGE')
                  for dataset in dataset_order]

fig = make_subplots(
    rows=rows, cols=cols,
    subplot_titles=subplot_titles,
    vertical_spacing=0.25,  # Increased from 0.15 to 0.25 to prevent overlap
    horizontal_spacing=0.1
)

# Color mapping based on demographic type
age_color = "#da81c1"  # Pink/purple for age
gender_color = "#7dbfa7"  # Teal/green for gender

# Track if we've added legend items
age_legend_added = False
gender_legend_added = False

# Add bars for each dataset
for idx, (dataset, groups) in enumerate(datasets_ordered.items(), 1):
    row = (idx - 1) // cols + 1
    col = (idx - 1) % cols + 1

    # Sort groups for consistent ordering
    sorted_groups = sorted(groups.items())
    group_names = [g[0] for g in sorted_groups]
    values = [g[1] for g in sorted_groups]

    # Assign colors based on original labels (before simplification)
    colors = []
    original_groups = datasets_original_labels_ordered[dataset]
    for display_label in group_names:
        # Find the original label that corresponds to this display label
        original_label = None
        for orig_label in original_groups.keys():
            if orig_label.split('_')[-1] == display_label or (orig_label == display_label):
                original_label = orig_label
                break

        if original_label and original_label.startswith('age'):
            colors.append(age_color)
        elif original_label and original_label.startswith('gender'):
            colors.append(gender_color)
        else:
            colors.append('gray')  # Fallback color

    # Determine text position and color based on relative bar height
    text_positions = []
    text_colors = []

    # Calculate the range of values for this dataset
    max_val = max(abs(v) for v in values) if values else 1

    for v in values:
        # Use relative threshold - if bar is less than 20% of the max, put text outside
        relative_threshold = 0.15 * max_val
        if abs(v) < relative_threshold:
            text_positions.append('outside')
            text_colors.append('black')
        else:
            text_positions.append('inside')
            text_colors.append('white')

    # Create bar trace with customized text (showing percentage)
    trace = go.Bar(
        x=group_names,
        y=values,
        name=dataset,
        marker_color=colors,
        text=[f'{v:.1f}' for v in values],  # Format as percentage with 1 decimal
        textposition=text_positions,
        textfont=dict(size=18, family="Computer Modern"),
        showlegend=False
    )

    # Update text colors (need to do this after adding trace)
    trace.textfont.color = text_colors

    fig.add_trace(trace, row=row, col=col)

    # Update axes with Computer Modern font and increased size
    fig.update_xaxes(
        tickangle=45,
        title_text="",  # Remove x-axis label
        row=row, col=col,
        tickfont=dict(size=22.5, family="Computer Modern", color="black"),  # 1.5x size
        tickcolor="black",
        linecolor="black"
    )
    fig.update_yaxes(
        title_text="F1 Difference (%)",
        row=row, col=col,
        tickfont=dict(size=22.5, family="Computer Modern", color="black"),  # 1.5x size
        title_font=dict(size=27, family="Computer Modern", color="black"),  # 1.5x size
        tickcolor="black",
        linecolor="black"
    )

    # Set custom y-axis range for ISIC2020 (first dataset in the reordered list)
    if dataset == 'isic2020':
        fig.update_yaxes(range=[-10, 30], row=row, col=col)  # Adjusted for percentage scale

# Add invisible traces for legend
fig.add_trace(
    go.Bar(
        x=[None],
        y=[None],
        name='Age',
        marker_color=age_color,
        showlegend=True
    )
)

fig.add_trace(
    go.Bar(
        x=[None],
        y=[None],
        name='Gender',
        marker_color=gender_color,
        showlegend=True
    )
)

# Update layout with white background and black text
fig.update_layout(
    height=750,  # 1.5x the original height
    width=1200,
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        font=dict(size=27, family="Computer Modern", color="black"),  # 1.5x size
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1
    ),
    margin=dict(t=100, b=50, l=50, r=50),
    font=dict(family="Computer Modern", color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Update subplot titles font
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=31.5, family="Computer Modern", color="black")  # 1.5x size

# Add horizontal line at y=0 for reference
# for i in range(1, rows + 1):
#     for j in range(1, cols + 1):
#         if (i - 1) * cols + j <= len(datasets_ordered):
#             fig.add_hline(y=0, line_color="light gray",
#                           opacity=0.5, row=i, col=j)

# Add gridlines for better readability
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

# Save as PDF
fig.write_image("fairgrpo_vs_grpo_f1_differences.pdf", engine="kaleido")
fig.show()

# Print summary statistics (excluding COVID-BLUES and aggregated metrics)
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

for dataset in dataset_order:
    print(f"\n{dataset}:")
    print("-" * 30)
    for group, diff in sorted(datasets_ordered[dataset].items()):
        sign = "+" if diff > 0 else ""
        # Find the original key to get raw values
        original_key = None
        for orig_label in datasets_original_labels_ordered[dataset].keys():
            if orig_label.split('_')[-1] == group or orig_label == group:
                # Reconstruct the full key
                if 'age' in orig_label:
                    original_key = f"fairness/{dataset}/age/{group}/f1"
                elif 'gender' in orig_label:
                    original_key = f"fairness/{dataset}/gender/{group}/f1"
                break
        
        if original_key and original_key in grpo_f1 and original_key in fairgrpo_f1:
            grpo_val = grpo_f1[original_key]
            fairgrpo_val = fairgrpo_f1[original_key]
            print(f"  {group:15s}: {sign}{diff:7.4f}, GRPO: {grpo_val:.4f}, FairGRPO: {fairgrpo_val:.4f}")

    # Dataset statistics
    values = list(datasets_ordered[dataset].values())
    if values:
        avg_diff = sum(values) / len(values)
        max_improvement = max(values)
        max_degradation = min(values)
        print(f"  {'Average':15s}: {avg_diff:+7.4f}")
        print(f"  {'Max Improvement':15s}: {max_improvement:+7.4f}")
        print(f"  {'Max Degradation':15s}: {max_degradation:+7.4f}")

print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)

all_diffs = []
for dataset in datasets_ordered.values():
    all_diffs.extend(dataset.values())

if all_diffs:
    print(f"Total groups analyzed: {len(all_diffs)}")
    print(f"Groups improved (diff > 0): {sum(1 for d in all_diffs if d > 0)}")
    print(f"Groups degraded (diff < 0): {sum(1 for d in all_diffs if d < 0)}")
    print(f"Groups unchanged (diff = 0): {sum(1 for d in all_diffs if d == 0)}")
    print(f"Average difference: {sum(all_diffs) / len(all_diffs):+.4f}")
    print(f"Max improvement: {max(all_diffs):+.4f}")
    print(f"Max degradation: {min(all_diffs):+.4f}")