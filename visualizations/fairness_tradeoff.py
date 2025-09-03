import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time

fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("random.pdf")
time.sleep(1)

# GRPO data (original)
steps_grpo = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
f1_scores_grpo = [0.18026521281980626, 0.2606731726246526, 0.25529697695262543,
                  0.24877426593441254, 0.24740053577229199, 0.24715526923171988,
                  0.24928431991602767, 0.25202210292384863, 0.2479242466439124,
                  0.24851959107431654, 0.25055224905394785, 0.2503139520742573,
                  0.2496623651606371, 0.24937904621940168]
accuracies_grpo = [0.6576341857656413, 0.739921065995784, 0.7772207126956295,
                   0.7936263776972936, 0.7967569967437476, 0.7983965638778587,
                   0.7994198671047513, 0.8003524126493133, 0.8003279159713251,
                   0.8001425648866994, 0.8019776277732922, 0.8031249037713727,
                   0.8034925489103545, 0.8032766351045837]
f1_diffs_grpo = [0.04688371312434529, 0.07199782262486236, 0.056119689057401036,
                 0.0668826276085771, 0.06941725227869947, 0.06520054185178974,
                 0.09932564932405598, 0.10523850548020404, 0.099675048968159,
                 0.09238919889312554, 0.08611769267733903, 0.10154214872103047,
                 0.09970283303993102, 0.10695660697340728]

# FairGRPO data (extended to step 90) - adding step 0 with same values as GRPO
steps_fairgrpo = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
f1_scores_fairgrpo = [0.18026521281980626, 0.23912621691116379, 0.2580314102296605, 0.24060319064057523,
                      0.24946177073251172, 0.25937497127430953, 0.260832953626606,
                      0.26886910626525184, 0.26325324146005696, 0.2628088308564845,
                      0.2601830174160277, 0.25775346825936163, 0.2604620181837937,
                      0.2729076044889513, 0.25294227803724506, 0.257977009756817,
                      0.2644689947013759, 0.25824138436715594, 0.2646768328130153]
accuracies_fairgrpo = [0.6576341857656413, 0.6972115797324159, 0.7700571127857955, 0.7885628840659245,
                       0.7959975892150549, 0.7994818994174401, 0.801384812383945,
                       0.8034552260354549, 0.8021861850362936, 0.8053813693013552,
                       0.8055253546084256, 0.8055758176380996, 0.8046618709085906,
                       0.8059481140565925, 0.8039018783207362, 0.8074242337969287,
                       0.8070877474372503, 0.8047067243248466, 0.8075475739502272]
f1_diffs_fairgrpo = [0.04688371312434529, 0.07610959233907032, 0.07305855610835758, 0.08008553494058117,
                     0.0557084348740649, 0.06558088066197464, 0.06753422373316542,
                     0.059125792613312855, 0.06266632024325304, 0.0671521083961389,
                     0.053515550125673446, 0.051869923113520615, 0.05311495633369936,
                     0.07247592620711198, 0.06580349339733502, 0.05153499404819173,
                     0.05215280732844841, 0.05035416356846994, 0.04519338120380582]

# Calculate training progress (0% to 100%)
max_step = 90
training_progress_grpo = [(step / max_step) * 100 for step in steps_grpo]
training_progress_fairgrpo = [(step / max_step) * 100 for step in steps_fairgrpo]

# For scatter plot, exclude the first point (step 0)
f1_scores_grpo_scatter = f1_scores_grpo[1:]
f1_diffs_grpo_scatter = f1_diffs_grpo[1:]
f1_scores_fairgrpo_scatter = f1_scores_fairgrpo[1:]
f1_diffs_fairgrpo_scatter = f1_diffs_fairgrpo[1:]

# Color palette
colors = ["#da81c1", "#7dbfa7", "#b0d766", "#8ca0cb", "#ee946c", "#da81c1"]
grpo_color = colors[0]
fairgrpo_color = colors[1]

# Standard layout settings
template = "plotly_white"
font_settings = dict(family="Computer Modern, serif", size=27, color="black")
tick_font = dict(size=23, color="black")
margin_settings = dict(l=10, r=10, t=10, b=10)

# Calculate combined ranges
all_f1 = f1_scores_grpo + f1_scores_fairgrpo
all_acc = accuracies_grpo + accuracies_fairgrpo
all_diff = f1_diffs_grpo + f1_diffs_fairgrpo

f1_min, f1_max = min(all_f1), max(all_f1)
f1_margin = (f1_max - f1_min) * 0.1
f1_range = [f1_min - f1_margin, f1_max + f1_margin]

acc_min, acc_max = min(all_acc), max(all_acc)
acc_margin = (acc_max - acc_min) * 0.1
acc_range = [acc_min - acc_margin, acc_max + acc_margin]

diff_min, diff_max = min(all_diff), max(all_diff)
diff_margin = (diff_max - diff_min) * 0.1
diff_range = [diff_min - diff_margin, diff_max + diff_margin]

# Create individual figures

# Plot 1: Training Progress vs F1 Score (4:3 aspect ratio)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=training_progress_grpo, y=f1_scores_grpo,
    mode='lines', name='GRPO',
    line=dict(color=grpo_color, width=5)
))
fig1.add_trace(go.Scatter(
    x=training_progress_fairgrpo, y=f1_scores_fairgrpo,
    mode='lines', name='FairGRPO',
    line=dict(color=fairgrpo_color, width=5)
))
fig1.update_layout(
    width=400, height=300, template=template, font=font_settings,
    xaxis_title="(a) Training Progress (%)", yaxis_title="F1 Score",
    xaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font),
    yaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font, range=f1_range),
    showlegend=False,
    margin=margin_settings
)

# Plot 2: Training Progress vs Accuracy (4:3 aspect ratio)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=training_progress_grpo, y=accuracies_grpo,
    mode='lines', name='GRPO',
    line=dict(color=grpo_color, width=5)
))
fig2.add_trace(go.Scatter(
    x=training_progress_fairgrpo, y=accuracies_fairgrpo,
    mode='lines', name='FairGRPO',
    line=dict(color=fairgrpo_color, width=5)
))
fig2.update_layout(
    width=400, height=300, template=template, font=font_settings,
    xaxis_title="(b) Training Progress (%)", yaxis_title="Accuracy",
    xaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font),
    yaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font, range=acc_range),
    legend=dict(orientation="v", yanchor="top", y=0.65, xanchor="right", x=0.99,
                font=dict(size=27, color="black"), bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1),
    margin=margin_settings
)

# Plot 3: Training Progress vs F1 Diff (4:3 aspect ratio)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=training_progress_grpo, y=f1_diffs_grpo,
    mode='lines', name='GRPO',
    line=dict(color=grpo_color, width=5)
))
fig3.add_trace(go.Scatter(
    x=training_progress_fairgrpo, y=f1_diffs_fairgrpo,
    mode='lines', name='FairGRPO',
    line=dict(color=fairgrpo_color, width=5)
))
fig3.update_layout(
    width=400, height=300, template=template, font=font_settings,
    xaxis_title="(c) Training Progress (%)", yaxis_title="F1 Diff",
    xaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font),
    yaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font, range=diff_range),
    showlegend=False,
    margin=margin_settings
)

# Runtime data for Plot 4
runtime_data = {
    'Qwen': {
        'FairGRPO': {'reward': 0.23, 'total': 886},
        'GRPO': {'reward': 0.20, 'total': 882}
    },
    'MedGemma': {
        'FairGRPO': {'reward': 0.61, 'total': 1388},
        'GRPO': {'reward': 0.44, 'total': 1371}
    }
}

# Plot 4: Runtime Comparison (4:3 aspect ratio) - restore normal range
models = ['Qwen', 'MedGemma']
bar_width = 0.23

x_positions = []
x_labels = []
bar_colors = []
bar_values = []
annotations = []

for i, model in enumerate(models):
    base_x = i * 1.0
    
    x_pos_total_grpo = base_x - 1.5 * bar_width
    x_positions.append(x_pos_total_grpo)
    x_labels.append('Total')
    bar_colors.append(grpo_color)
    bar_values.append(runtime_data[model]['GRPO']['total'])
    annotations.append((x_pos_total_grpo, runtime_data[model]['GRPO']['total'], f"{runtime_data[model]['GRPO']['total']}", "white"))
    
    x_pos_reward_grpo = base_x - 0.5 * bar_width
    x_positions.append(x_pos_reward_grpo)
    x_labels.append('Advantage')
    bar_colors.append(grpo_color)
    bar_values.append(runtime_data[model]['GRPO']['reward'])
    annotations.append((x_pos_reward_grpo, runtime_data[model]['GRPO']['reward'], f"{runtime_data[model]['GRPO']['reward']:.2f}", "black"))
    
    x_pos_total_fairgrpo = base_x + 0.5 * bar_width
    x_positions.append(x_pos_total_fairgrpo)
    x_labels.append('Total')
    bar_colors.append(fairgrpo_color)
    bar_values.append(runtime_data[model]['FairGRPO']['total'])
    annotations.append((x_pos_total_fairgrpo, runtime_data[model]['FairGRPO']['total'], f"{runtime_data[model]['FairGRPO']['total']}", "white"))
    
    x_pos_reward_fairgrpo = base_x + 1.5 * bar_width
    x_positions.append(x_pos_reward_fairgrpo)
    x_labels.append('Advantage')
    bar_colors.append(fairgrpo_color)
    bar_values.append(runtime_data[model]['FairGRPO']['reward'])
    annotations.append((x_pos_reward_fairgrpo, runtime_data[model]['FairGRPO']['reward'], f"{runtime_data[model]['FairGRPO']['reward']:.2f}", "black"))

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=x_positions, y=bar_values,
    marker_color=bar_colors, width=bar_width
))

# Add annotations
for x_pos, bar_val, text, color in annotations:
    if color == "white":
        y_pos = bar_val * 0.5
    else:
        y_pos = bar_val + max(bar_values) * 0.15
    
    fig4.add_annotation(
        x=x_pos, y=y_pos, text=text,
        showarrow=False, textangle=90,
        font=dict(color=color, size=16, family="Computer Modern, serif")
    )

# Add model labels (normal range, not constrained)
for i, model in enumerate(models):
    fig4.add_annotation(
        x=i * 0.9, y=max(bar_values) * 1.15,
        text=model, showarrow=False,
        font=dict(color="black", size=20, family="Computer Modern, serif")
    )

fig4.update_layout(
    width=400, height=300, template=template, font=font_settings,
    xaxis_title="(d) Runtime Per Step (s)", yaxis_title="Runtime (s)",
    xaxis=dict(
        tickvals=x_positions, ticktext=x_labels * len(models),
        titlefont=dict(size=27, color="black"), tickfont=dict(size=18, color="black"),
        tickangle=-45
    ),
    yaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font),
    showlegend=False, margin=margin_settings
)

# Scatter plot - Convert F1 diff to 1 - F1 diff so higher is better
f1_diffs_grpo_inverted = [1 - diff for diff in f1_diffs_grpo_scatter]
f1_diffs_fairgrpo_inverted = [1 - diff for diff in f1_diffs_fairgrpo_scatter]

# Calculate axis ranges for scatter plot
scatter_f1 = f1_scores_grpo_scatter + f1_scores_fairgrpo_scatter
scatter_diff_inverted = f1_diffs_grpo_inverted + f1_diffs_fairgrpo_inverted
plot_x_min, plot_x_max = min(scatter_f1), max(scatter_f1)
plot_y_min, plot_y_max = min(scatter_diff_inverted), max(scatter_diff_inverted)
x_margin = (plot_x_max - plot_x_min) * 0.1
y_margin = (plot_y_max - plot_y_min) * 0.1
axis_x_min = plot_x_min - x_margin
axis_x_max = plot_x_max + x_margin  
axis_y_min = plot_y_min - y_margin
axis_y_max = plot_y_max + y_margin

# Find Pareto frontiers
def find_pareto_frontier_with_axis_ranges(x_values, y_values, axis_x_min, axis_x_max, axis_y_min, axis_y_max):
    points = list(zip(x_values, y_values))
    
    pareto_points = []
    for i, (x1, y1) in enumerate(points):
        is_dominated = False
        for j, (x2, y2) in enumerate(points):
            if i != j and x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append((x1, y1))
    
    filtered_frontier = []
    max_y_seen = float('inf')
    for x, y in sorted(pareto_points, key=lambda p: p[0]):
        if y < max_y_seen:
            filtered_frontier.append((x, y))
            max_y_seen = y
    
    extended_frontier = []
    if filtered_frontier:
        leftmost_point = filtered_frontier[0]
        rightmost_point = filtered_frontier[-1]
        extended_frontier.append((axis_x_min, leftmost_point[1]))
        extended_frontier.extend(filtered_frontier)
        extended_frontier.append((rightmost_point[0], axis_y_min))
    else:
        best_point = max(points, key=lambda p: p[1])
        extended_frontier = [(axis_x_min, best_point[1]), best_point, (best_point[0], axis_y_min)]
    
    return extended_frontier

grpo_frontier = find_pareto_frontier_with_axis_ranges(f1_scores_grpo_scatter, f1_diffs_grpo_inverted, 
                                                     axis_x_min, axis_x_max, axis_y_min, axis_y_max)
fairgrpo_frontier = find_pareto_frontier_with_axis_ranges(f1_scores_fairgrpo_scatter, f1_diffs_fairgrpo_inverted,
                                                         axis_x_min, axis_x_max, axis_y_min, axis_y_max)

# Scatter plot (square - width=height, 2x size = 600x600)
fig_scatter = go.Figure()

# Add Pareto frontiers
if grpo_frontier:
    frontier_x = [p[0] for p in grpo_frontier]
    frontier_y = [p[1] for p in grpo_frontier]
    fig_scatter.add_trace(go.Scatter(
        x=frontier_x, y=frontier_y, mode='lines',
        name='GRPO Frontier', line=dict(color=grpo_color, width=2, dash='dash'),
        showlegend=False
    ))

if fairgrpo_frontier:
    frontier_x = [p[0] for p in fairgrpo_frontier]
    frontier_y = [p[1] for p in fairgrpo_frontier]
    fig_scatter.add_trace(go.Scatter(
        x=frontier_x, y=frontier_y, mode='lines',
        name='FairGRPO Frontier', line=dict(color=fairgrpo_color, width=2, dash='dash'),
        showlegend=False
    ))

# Add scatter points
fig_scatter.add_trace(go.Scatter(
    x=f1_scores_grpo_scatter, y=f1_diffs_grpo_inverted,
    mode='markers', name='GRPO',
    marker=dict(color=grpo_color, size=15, line=dict(color='white', width=1))
))
fig_scatter.add_trace(go.Scatter(
    x=f1_scores_fairgrpo_scatter, y=f1_diffs_fairgrpo_inverted,
    mode='markers', name='FairGRPO',
    marker=dict(color=fairgrpo_color, size=15, line=dict(color='white', width=1))
))

# Add best points (stars)
fig_scatter.add_trace(go.Scatter(
    x=[f1_scores_grpo_scatter[0]], y=[f1_diffs_grpo_inverted[0]],
    mode='markers', name='Best GRPO',
    marker=dict(color=grpo_color, size=24, symbol='star', line=dict(color='black', width=2)),
    showlegend=False
))
fig_scatter.add_trace(go.Scatter(
    x=[f1_scores_fairgrpo_scatter[-1]], y=[f1_diffs_fairgrpo_inverted[-1]],
    mode='markers', name='Best FairGRPO',
    marker=dict(color=fairgrpo_color, size=24, symbol='star', line=dict(color='black', width=2)),
    showlegend=False
))

fig_scatter.update_layout(
    width=600, height=600, template=template, font=font_settings,  # Square, 2x size
    xaxis_title="(e) F1 Score", yaxis_title="1 - F1 Diff",
    xaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font,
               range=[min(scatter_f1) - (max(scatter_f1) - min(scatter_f1)) * 0.1,
                      max(scatter_f1) + (max(scatter_f1) - min(scatter_f1)) * 0.1]),
    yaxis=dict(titlefont=dict(size=27, color="black"), tickfont=tick_font),
    legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.99,
                font=dict(size=27, color="black"), bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1),
    margin=margin_settings
)

# Save all figures in PDF and PNG, don't show
fig1.write_image("plot1_f1.pdf", width=400, height=300, scale=3)
fig1.write_image("plot1_f1.png", width=400, height=300, scale=3)

fig2.write_image("plot2_accuracy.pdf", width=400, height=300, scale=3) 
fig2.write_image("plot2_accuracy.png", width=400, height=300, scale=3)

fig3.write_image("plot3_f1diff.pdf", width=400, height=300, scale=3)
fig3.write_image("plot3_f1diff.png", width=400, height=300, scale=3)

fig4.write_image("plot4_runtime.pdf", width=400, height=300, scale=3)
fig4.write_image("plot4_runtime.png", width=400, height=300, scale=3)

fig_scatter.write_image("plot5_scatter.pdf", width=600, height=600, scale=3)
fig_scatter.write_image("plot5_scatter.png", width=600, height=600, scale=3)

print("Individual plots saved:")
print("- plot1_f1: F1 Score vs Training Progress (1200x900 pixels)")
print("- plot2_accuracy: Accuracy vs Training Progress (1200x900 pixels)")  
print("- plot3_f1diff: F1 Diff vs Training Progress (1200x900 pixels)")
print("- plot4_runtime: Runtime comparison with model labels (1200x900 pixels)")
print("- plot5_scatter: Performance-fairness scatter plot (1800x1800 pixels)")
print("All saved in both PDF and PNG formats")