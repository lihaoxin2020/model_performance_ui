import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from io import StringIO

# Utility: create bar plot given dataframe, score column, filename
def vendor_barplot(df, value_col, title, ylabel, fname):
    # Model display name mapping - Compact names for 2-column layout
    model_display_names = {
        "deepseek-v3": "DeepSeek-V3",
        "gemini2.5-pro-high": "Gemini 2.5 (H)",
        "o3-low": "o3 (L)",
        "Qwen3-32B": "Qwen3-32B",
        "claude-sonnet-4-low": "Claude 4 (L)",
        "Qwen3-32B-thinking": "Qwen3-32B (T)",
        "gemini-2.5-pro-low": "Gemini 2.5 (L)",
        "gpt-4.1": "GPT-4.1",
        "deepseek-r1": "DeepSeek-R1",
        "claude-sonnet-4-high": "Claude 4 (H)",
        "o3-high": "o3 (H)",
        "deepseek-r1-05": "DeepSeek-R1-05",
        "o3-mini-high": "o3-mini (H)",
        "o3-mini-low": "o3-mini (L)",
        "gemini-2.5-pro-preview-05-06-low": "Gemini 2.5 (L)",
        "o4-mini-low": "o4-mini (L)",
        "o4-mini-high": "o4-mini (H)"
    }
    
    # Vendor mapping (color, icon text)
    vendor_info = {
        "OpenAI": {"color": "#000000", "text": "OA"},
        "Google": {"color": "#34a853", "text": "G"},
        "DeepSeek": {"color": "#3366cc", "text": "DS"},
        "Anthropic": {"color": "#e67e22", "text": "A"},
        "Qwen": {"color": "#888888", "text": "O"},
    }
    
    # Reasoning effort mapping (hatching patterns)
    effort_patterns = {
        "low": "///",      # diagonal lines
        "high": "",     # dots
        "mini": "",     # crosses
        "preview": "",  # horizontal lines
        "thinking": "", # vertical lines
        "r1": "",       # plus signs
        "base": "///"         # no pattern (solid)
    }
    
    # Identify vendor per model
    def get_vendor(model):
        m = model.lower()
        if m.startswith("o3") or m.startswith("o4") or m.startswith("gpt"):
            return "OpenAI"
        if "gemini" in m:
            return "Google"
        if m.startswith("deepseek"):
            return "DeepSeek"
        if "claude" in m:
            return "Anthropic"
        if "qwen" in m:
            return "Qwen"
        return "Other"
    
    # Identify reasoning effort level
    def get_effort_pattern(model):
        m = model.lower()
        if "-low" in m:
            return "low"
        elif "-high" in m:
            return "high"
        elif "mini" in m:
            return "mini"
        elif "preview" in m:
            return "preview"
        elif "thinking" in m:
            return "thinking"
        elif "r1" in m:
            return "r1"
        else:
            return "base"
    
    df["vendor"] = df["model"].apply(get_vendor)
    df["effort"] = df["model"].apply(get_effort_pattern)
    df["color"] = df["vendor"].apply(lambda v: vendor_info[v]["color"])
    df["hatch"] = df["effort"].apply(lambda e: effort_patterns[e])
    
    # Apply display name mapping
    df["display_name"] = df["model"].apply(lambda x: model_display_names.get(x, x))
    
    df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    
    # Plot - Compact size for 2-column academic paper
    fig, ax = plt.subplots(figsize=(6, 4))  # Much more compact dimensions
    bars = ax.bar(df["display_name"], df[value_col], color=df["color"], hatch=df["hatch"], edgecolor='white', linewidth=0.5, width=0.7)  # Slightly narrower bars
    
    # Annotate value on each bar with improved visibility
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Place text at the center of the bar
        ax.text(bar.get_x()+bar.get_width()/2, height/2,
                f"{height*100:.1f}",
                ha='center', va='center', fontsize=8, color='white', weight='bold')  # Smaller font for compact display
    
    # Add circular icon under each bar - Smaller for compact display
    y_icon = -0.02 * df[value_col].max()  # small negative offset
    for i, row in df.iterrows():
        x_center = bars[i].get_x() + bars[i].get_width()/2
        circ = Circle((x_center, y_icon), 0.012, color=row["color"], transform=ax.get_xaxis_transform())  # Smaller circle
        ax.add_patch(circ)
        ax.text(x_center, y_icon, vendor_info[row["vendor"]]["text"],
                ha='center', va='center', color='white', fontsize=5, transform=ax.get_xaxis_transform())  # Smaller text
    
    # Style improvements for academic paper
    ax.set_ylabel(ylabel, fontsize=10, weight='bold')  # Smaller font for compact display
    
    # Adjust y-axis limits based on the data range for better spacing
    if title and title.startswith("Overall"):
        # For overall plot, use wider range for better bar spacing visualization
        ax.set_ylim(0, max(0.8, df[value_col].max()*1.2))
    else:
        ax.set_ylim(0, df[value_col].max())
    
    if title:
        ax.set_title(title, fontsize=12, weight='bold', pad=15)  # Smaller font and padding for compact display
    
    # Fix tick labels warning by setting tick positions first
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["display_name"], rotation=45, ha='right', fontsize=8)  # More vertical rotation and smaller font
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=9)  # Slightly smaller y-axis labels
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Create legends
    # Vendor legend
    vendor_handles = []
    for vendor, info in vendor_info.items():
        if vendor in df["vendor"].values:
            vendor_handles.append(plt.Rectangle((0,0),1,1, facecolor=info["color"], label=vendor))
    
    # Effort level legend
    effort_handles = []
    effort_labels = {
        "low": "Non-reasoning/Low effort",
        "high": "Thinking/High effort", 
        # "mini": "Mini model",
        # "preview": "Preview",
        "thinking": "Thinking/High effort",
        "r1": "Thinking/High effort",
        "base": "Non-reasoning/Low effort"
    }
    # effort_labels = {
    #     "low": "///",      # diagonal lines
    #     "high": "",     # dots
    #     "mini": "",     # crosses
    #     "preview": "",  # horizontal lines
    #     "thinking": "", # vertical lines
    #     "r1": "",       # plus signs
    #     "base": "///"         # no pattern (solid)
    # }
    existing_efforts = []
    for effort in df["effort"].unique():
        if effort in effort_labels and effort_labels[effort] not in existing_efforts:
            existing_efforts.append(effort_labels[effort])
            effort_handles.append(plt.Rectangle((0,0),1,1, facecolor='gray', hatch=effort_patterns[effort], 
                                            edgecolor='white', linewidth=0.5, label=effort_labels[effort]))
    
    # Add merged legend - Single legend box for compact layout
    combined_handles = vendor_handles.copy()
    if effort_handles:
        combined_handles.extend(effort_handles)
    
    # Create labels that include both vendor and effort info
    combined_labels = [handle.get_label() for handle in combined_handles]
    
    # Single merged legend positioned centrally
    legend = ax.legend(handles=combined_handles, labels=combined_labels,
                      loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=7, 
                      ncol=min(len(combined_handles), 4),  # Max 4 columns to avoid overcrowding
                      title="Model Provider & Reasoning Effort", title_fontsize=8, 
                      frameon=True, fancybox=True, shadow=True)
    legend.get_title().set_fontweight('bold')
    
    # Hide outer frame (spines) for cleaner academic look
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Keep only left spine for y-axis reference
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#666666')
    
    plt.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.show()
    return fname

# ---- SciReasBench‑Pro dataset ----
scireas_raw = """model score
deepseek-v3 0.303968254
gemini2.5-pro-high 0.6357142857
o3-low 0.530952381
Qwen3-32B 0.2023809524
claude-sonnet-4-low 0.3293650794
Qwen3-32B-thinking 0.4682539683
gemini-2.5-pro-low 0.6126984127
gpt-4.1 0.2711213518
deepseek-r1 0.5031746032
claude-sonnet-4-high 0.3873015873
o3-high 0.6468253968
deepseek-r1-05 0.4430894309
"""
df_scireas = pd.read_csv(StringIO(scireas_raw), sep=" ")

scireas_path = "./scireas_vendor_bar.png"
vendor_barplot(df_scireas, "score",
               title=None,
               ylabel="Micro‑avg", 
               fname=scireas_path)

# ---- Overall composite avg ----
overall_raw = """model avg
deepseek-v3 0.5791978352
gemini2.5-pro-high 0.6698512344
o3-low 0.6615495159
Qwen3-32B 0.5143679836
o3-mini-high 0.637433532
claude-sonnet-4-low 0.6027870532
Qwen3-32B-thinking 0.6198301754
o3-mini-low 0.5691501205
gemini-2.5-pro-preview-05-06-low 0.6664386217
gpt-4.1 0.5871072802
deepseek-r1 0.6416857088
claude-sonnet-4-high 0.6212321219
o3-high 0.6836264792
deepseek-r1-05 0.6509411591
o4-mini-low 0.6130330731
o4-mini-high 0.6536338486
"""
df_overall = pd.read_csv(StringIO(overall_raw), sep=" ")

overall_path = "./overall_vendor_bar.png"
vendor_barplot(
    df_overall, "avg",
    title=None,
    ylabel="SciReasBench Avg", 
    fname=overall_path
)

(scireas_path, overall_path)
