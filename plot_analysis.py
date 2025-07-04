import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from io import StringIO
from matplotlib.lines import Line2D

# Re‑use the same raw table
raw_data = """model gpqa lab_bench mmlu olympiadbench scibench scieval sciknoweval sciriff supergpqa ugphysics avg
deepseek-v3 0.625 0.5601571268 0.8385714286 0.5116934064 0.6632947977 0.8410057471 0.4467443031 0.5244384075 0.3170411125 0.4640320226 0.5791978352
gemini2.5-pro-high 0.7946428571 0.6443714927 0.8621428571 0.6955187849 0.7023121387 0.851091954 0.4760665602 0.5138065074 0.6044812872 0.554077905 0.6698512344
o3-low 0.7544642857 0.7050841751 0.8571428571 0.5352122416 0.6965317919 0.8483333333 0.5211024387 0.5176356024 0.5490925479 0.6308958847 0.6615495159
Qwen3-32B 0.4888392857 0.4456116723 0.7871428571 0.334538299 0.5939306358 0.8350862069 0.4268418753 0.5159763714 0.3097608638 0.4059517689 0.5143679836
o3-mini-high 0.7388392857 0.5921997755 0.85 0.5111778907 0.6632947977 0.8343390805 0.5192469325 0.5183971394 0.5399058742 0.6069345443 0.637433532
claude-sonnet-4-low 0.6383928571 0.5337654321 0.8414285714 0.5541668762 0.6546242775 0.8575862069 0.4361794175 0.5351842317 0.4522036813 0.5243389805 0.6027870532
Qwen3-32B-thinking 0.6383928571 0.4801627385 0.8542857143 0.5208218076 0.7268786127 0.8359195402 0.439401722 0.5065375723 0.4926573169 0.7032438727 0.6198301754
o3-mini-low 0.6339285714 0.569349046 0.8214285714 0.3950485339 0.4595375723 0.8376724138 0.4895141603 0.5128561696 0.4049072958 0.5672588702 0.5691501205
gemini-2.5-pro-preview-05-06-low 0.8013392857 0.6189842873 0.85 0.6746466831 0.7095375723 0.8643390805 0.4681291809 0.515978676 0.6010012297 0.5604302216 0.6664386217
gpt-4.1 0.6183035714 0.5854601571 0.83 0.4272431223 0.6575144509 0.8683333333 0.4856699628 0.5307850872 0.3623627962 0.5054003204 0.5871072802
deepseek-r1 0.7232142857 0.6073961841 0.8685714286 0.5213498969 0.7066473988 0.8609195402 0.4603416848 0.5221190946 0.5705987262 0.5756988486 0.6416857088
claude-sonnet-4-high 0.6897321429 0.571969697 0.8528571429 0.5983503495 0.6705202312 0.8576724138 0.4326906391 0.5092996968 0.49763368 0.5315952257 0.6212321219
o3-high 0.7991071429 0.7417901235 0.8664285714 0.5797100538 0.7210982659 0.8267528736 0.5194663307 0.5355263259 0.5947069858 0.6516781184 0.6836264792
deepseek-r1-05 0.7232142857 0.599315376 0.86 0.6238872404 0.6994219653 0.8592528736 0.4621593143 0.5080238246 0.6026804852 0.5714562265 0.6509411591
o4-mini-low 0.6941964286 0.5972334456 0.8407142857 0.4041643615 0.6546242775 0.8709195402 0.4992566449 0.506047602 0.4863251541 0.5768489907 0.6130330731
o4-mini-high 0.7455357143 0.6365937149 0.86 0.4963473822 0.6965317919 0.875 0.5105978871 0.5218619818 0.5713918718 0.6224781422 0.6536338486
"""
# Build DataFrame
lines = raw_data.strip().split("\n")
cols = lines[0].split()
data = []
for line in lines[1:]:
    parts = line.split()
    model = parts[0]
    vals = list(map(float, parts[1:]))
    data.append([model] + vals)
df = pd.DataFrame(data, columns=cols)

score_cols = [c for c in df.columns if c not in ("model", "avg")]

# Prepare SciReasBench data
raw_full = """model avg
deepseek-v3 0.5791978352
gemini2.5-pro-high 0.6698512344
o3-low 0.6615495159
Qwen3-32B 0.5143679836
claude-sonnet-4-low 0.6027870532
Qwen3-32B-thinking 0.6198301754
gemini-2.5-pro-preview-05-06-low 0.6664386217
gpt-4.1 0.5871072802
deepseek-r1 0.6416857088
claude-sonnet-4-high 0.6212321219
o3-high 0.6836264792
deepseek-r1-05 0.6509411591
"""

df_avg = pd.read_csv(StringIO(raw_full), sep=" ")

scireas_raw = """model scireas
deepseek-v3 0.303968254
gemini2.5-pro-high 0.6357142857
o3-low 0.530952381
Qwen3-32B 0.2023809524
claude-sonnet-4-low 0.3293650794
Qwen3-32B-thinking 0.4682539683
gemini-2.5-pro-preview-05-06-low 0.6126984127
gpt-4.1 0.2711213518
deepseek-r1 0.5031746032
claude-sonnet-4-high 0.3873015873
o3-high 0.6468253968
deepseek-r1-05 0.4430894309
"""

scireas_df = pd.read_csv(StringIO(scireas_raw), sep=" ")
merged = pd.merge(df_avg, scireas_df, on="model", how="inner")

# Calculate correlations for subplots (b) and (c)
corr_scibench = df["avg"].corr(df["scibench"])
corr_mmlu = df["avg"].corr(df["mmlu"])

# -------------- Create single figure with 3 horizontal subplots -----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Create color mapping for models
unique_models = df["model"].unique()
colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_models)))
model_colors = dict(zip(unique_models, colors))

# Create mapping from technical names to display names
model_display_names = {
    # o3 family
    "o3-high": "o3 (High)",
    "o3-low": "o3 (Low)",
    "o3-mini-high": "o3-Mini (High)", 
    "o3-mini-low": "o3-Mini (Low)",
    
    # o4 family
    "o4-mini-high": "o4-Mini (High)",
    "o4-mini-low": "o4-Mini (Low)",
    
    # DeepSeek family
    "deepseek-v3": "DeepSeek-V3",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-r1-05": "DeepSeek-R1-0528",
    
    # Gemini family
    "gemini2.5-pro-high": "Gemini-2.5-Pro (High)",
    "gemini-2.5-pro-preview-05-06-low": "Gemini-2.5-Pro (Low)",
    
    # Claude family
    "claude-sonnet-4-high": "Claude-Sonnet-4 (High)",
    "claude-sonnet-4-low": "Claude-Sonnet-4 (Low)",
    
    # Qwen family
    "Qwen3-32B": "Qwen3-32B",
    "Qwen3-32B-thinking": "Qwen3-32B (Thinking)",
    
    # GPT family
    "gpt-4.1": "GPT-4.1"
}

# Subplot 1: Correlation Heatmap
corr = df[score_cols].corr()
im = axes[0].imshow(corr.values, cmap='viridis')
axes[0].set_xticks(np.arange(len(score_cols)))
axes[0].set_yticks(np.arange(len(score_cols)))
axes[0].set_xticklabels(score_cols, rotation=45, ha="right", fontsize=12)
axes[0].set_yticklabels(score_cols, fontsize=12)

# Annotate each cell with the value
for i in range(len(score_cols)):
    for j in range(len(score_cols)):
        axes[0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=10, color="white")

axes[0].set_title("(a) SciReasBench Correlation", fontsize=16, pad=15)

# Subplot 2: Overall Average vs SciBench (with color coding and correlation)
for i, txt in enumerate(df["model"]):
    axes[1].scatter(df["avg"][i], df["scibench"][i], s=80, alpha=0.8, 
                   color=model_colors[txt])
axes[1].set_xlabel("SciReasBench Average", fontsize=14)
axes[1].set_ylabel("SciBench Score", fontsize=14)
axes[1].set_title(f"(b)SciReasBench vs SciBench(ρ = {corr_scibench:.2f})", fontsize=16)
axes[1].grid(True, linestyle=":", alpha=0.5)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# Subplot 3: Overall Average vs MMLU-Pro (with color coding and correlation)
for i, txt in enumerate(df["model"]):
    axes[2].scatter(df["avg"][i], df["mmlu"][i], s=80, alpha=0.8, 
                   color=model_colors[txt])
axes[2].set_xlabel("SciReasBench Average", fontsize=14)
axes[2].set_ylabel("MMLU-Pro* Score", fontsize=14)
axes[2].set_title(f"(c)SciReasBench vs MMLU-Pro* (ρ = {corr_mmlu:.2f})", fontsize=16)
axes[2].grid(True, linestyle=":", alpha=0.5)
axes[2].tick_params(axis='both', which='major', labelsize=12)

# Add a shared legend on the right side
# Create handles and labels in model_display_names order
handles = []
display_labels = []
for model in model_display_names:
    if model in model_colors:
        handles.append(Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=model_colors[model],
                                markersize=10, label=model_display_names[model], 
                                alpha=0.8))
        display_labels.append(model_display_names[model])

fig.legend(handles, display_labels, loc='center left',
          bbox_to_anchor=(0.82, 0.5), ncol=1, fontsize=12)

# Adjust layout with reduced spacing and save
plt.subplots_adjust(wspace=0.3, right=0.82)  # Add right margin for legend
combined_path = "./combined_analysis_figure.png"
fig.savefig(combined_path, dpi=300, bbox_inches='tight')

plt.show()

print(f"Saved combined figure: {combined_path}")
print(f"Correlation SciBench: {corr_scibench:.3f}")
print(f"Correlation MMLU-Pro: {corr_mmlu:.3f}")

# Optional: Save ranking shift table for reference
merged["rank_avg"] = merged["avg"].rank(ascending=False, method='min')
merged["rank_scireas"] = merged["scireas"].rank(ascending=False, method='min')
merged["shift"] = merged["rank_avg"] - merged["rank_scireas"]

shift_table = merged[["model","rank_avg","rank_scireas","shift"]].sort_values(by="rank_scireas")
shift_path = "./rank_shifts.csv"
shift_table.to_csv(shift_path, index=False)

print(f"Ranking shift analysis saved: {shift_path}")
print("\nFirst few rows of ranking shifts:")
print(shift_table.head().to_string(index=False))