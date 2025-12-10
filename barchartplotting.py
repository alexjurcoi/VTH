import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch

#sns.set_theme(style="whitegrid", palette="viridis")

FREQ_COLORS = {
    20: "tab:orange",   # orange
    200: "red"          # red
}

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# ---------- helpers ----------
def list_excel_files(folder="."):
    return [f for f in os.listdir(folder) if f.lower().endswith(".xlsx")]

def parse_metadata_from_filename(fname):
    meta = {"file": fname}
    v_match = re.search(r"_V_(-?\d+\.\d+)", fname)
    if v_match:
        meta["V"] = float(v_match.group(1))
    dg_match = re.search(r"_dG_([0-9\.]+)-([0-9\.]+)eV", fname)
    if dg_match:
        meta["dGmin"] = float(dg_match.group(1))
        meta["dGmax"] = float(dg_match.group(2))
    f_match = re.search(r"freq_([0-9eE\+\-\.]+(?:-[0-9eE\+\-\.]+)*)", fname)
    if f_match:
        try:
            meta["freqs"] = [float(f) for f in f_match.group(1).split("-")]
        except ValueError:
            meta["freqs"] = []
    else:
        meta["freqs"] = []
    return meta


def plot_bar_comparison(all_freqs, all_avg_rT, static_rT,
                        target_V, target_dGmin, target_dGmax, folder):

    # ---- sort frequencies ----
    all_freqs, all_avg_rT = zip(*sorted(zip(all_freqs, all_avg_rT), key=lambda x: x[0]))
    freqs = np.array(all_freqs, dtype=float)
    dyn = np.array(all_avg_rT, dtype=float)
    stat = np.full_like(dyn, float(static_rT))

    # ---- x positions and labels ----
    x = np.arange(len(freqs))
    # Use plain integer formatting (no scientific notation)
    labels = [str(int(f)) for f in freqs]


    fig, ax = plt.subplots(figsize=(8, 5), dpi=400)
    width = 0.38

    # ---- dynamic bars ----
    dyn_colors = [FREQ_COLORS.get(int(round(f)), "#b22222") for f in freqs]
    bd = ax.bar(x - width/2, dyn, width=width, edgecolor="black", linewidth=1.2)
    for i, b in enumerate(bd):
        b.set_facecolor(dyn_colors[i])
        b.set_hatch("//")

    # ---- static bars ----
    bs = ax.bar(x + width/2, stat, width=width, color="#a9a9a9",
                edgecolor="black", linewidth=1.2)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    # ---- labels and title ----
    ax.set_xticks(x, labels)
    ax.set_xlabel("Frequency (Hz)", labelpad=10)
    ax.set_ylabel("Average $r_T$ (mol/cm²·s)", labelpad=10)
    ax.set_title(
        f"Average $r_T$ Comparison by Frequency\n"
        f"(V = {target_V:.2f} V, ΔG = {target_dGmin:.2f}–{target_dGmax:.2f} eV)",
        pad=15
    )

    # ---- y limits ----
    vals = np.concatenate([dyn, stat])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    pad = 0.25 * (vmax - vmin)
    ax.set_ylim(vmin - pad, vmax + pad)
    
    # ---- single, clean annotation set ----
    for b in list(bd) + list(bs):
        h = b.get_height()
        if h >= 0:
            y_text = h + 0.02 * (vmax - vmin)  # fixed offset above
            va = "bottom"
        else:
            y_text = h - 0.02 * (vmax - vmin)  # fixed offset below
            va = "top"
        ax.text(
            b.get_x() + b.get_width() / 2,
            y_text,
            f"{abs(h):.2e}",
            ha="center",
            va=va,
            fontsize=12,
            fontweight="bold",
            color="black"
        )
        
# =============================================================================
#         # ---- % change annotations ----
#     for i, (dyn_val, stat_val) in enumerate(zip(dyn, stat)):
#         if stat_val == 0 or np.isnan(stat_val):
#             continue
#         pct_change = 100 * (dyn_val - stat_val) / stat_val
#     
#         # Place annotation roughly above both bars
#         y_max = max(dyn_val, stat_val)
#         y_text = y_max + 0.10 * (vmax - vmin)
#     
#         ax.text(
#             x[i],
#             y_text,
#             f"{pct_change:+.1f}%",
#             ha="center",
#             va="bottom",
#             fontsize=13,
#             fontweight="bold",
#             color="green" if pct_change > 0 else "red",
#         )
# 
# 
# =============================================================================
    # ---- split-color dynamic legend ----
    class DualColorPatch(Patch):
        def __init__(self, color_left, color_right, **kwargs):
            super().__init__(**kwargs)
            self.color_left = color_left
            self.color_right = color_right

    class HandlerDualColor(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            left = mpatches.Rectangle(
                (xdescent, ydescent), width / 2, height,
                facecolor=orig_handle.color_left,
                edgecolor='black', hatch='//', lw=1.0, transform=trans)
            right = mpatches.Rectangle(
                (xdescent + width / 2, ydescent), width / 2, height,
                facecolor=orig_handle.color_right,
                edgecolor='black', hatch='//', lw=1.0, transform=trans)
            return [left, right]

    color_left, color_right = FREQ_COLORS[20], FREQ_COLORS[200]
    dynamic_patch = DualColorPatch(color_left, color_right, label="Dynamic")
    static_patch = mpatches.Patch(facecolor="#a9a9a9", edgecolor="black", label="Static")

    # ---- legend positioned close to plot ----
    ax.legend(
        handles=[dynamic_patch, static_patch],
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),  # closer than before
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        fontsize=12,
        handlelength=2.2,
        handler_map={DualColorPatch: HandlerDualColor()}
    )

    ax.tick_params(axis="both", which="major", width=1.2, length=6)
    #ax.set_ylim(-5e-7, 6e-7)
    ax.grid(False)

    plt.tight_layout(rect=[0, 0, 0.90, 1])  # allocate space for legend
    outname = os.path.join(
        folder,
        f"avg_rT_bar_vs_freq_V{target_V:.2f}_dG{target_dGmin:.2f}-{target_dGmax:.2f}.png"
    )
    plt.savefig(outname, dpi=400, bbox_inches="tight")
    print(f"✅ Saved clean chart → {outname}")
    plt.show()


# ---------- main ----------
def main():
    folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
    files = list_excel_files(folder)
    if not files:
        print("No .xlsx files found.")
        return

    metadata = [parse_metadata_from_filename(f) for f in files]

    # Voltage selection
    voltages = sorted({m["V"] for m in metadata if "V" in m})
    print("\nAvailable voltages:")
    for i, V in enumerate(voltages, 1):
        print(f"{i}: {V} V")
    k = int(input("Pick voltage #: ").strip())
    target_V = voltages[k - 1]

    # Oscillation window selection
    dg_windows = sorted({(m["dGmin"], m["dGmax"]) for m in metadata})
    print("\nAvailable oscillation windows (ΔG ranges):")
    for i, (dmin, dmax) in enumerate(dg_windows, 1):
        print(f"{i}: {dmin:.2f} – {dmax:.2f} eV")
    k = int(input("Pick oscillation window #: ").strip())
    target_dGmin, target_dGmax = dg_windows[k - 1]

    # Filter files
    group = [
        m for m in metadata
        if np.isclose(m["V"], target_V)
        and np.isclose(m["dGmin"], target_dGmin)
        and np.isclose(m["dGmax"], target_dGmax)
    ]

    all_freqs, all_avg_rT = [], []
    static_rT = None

    for m in group:
        filepath = os.path.join(folder, m["file"])
        try:
            xls = pd.ExcelFile(filepath)
        except Exception as e:
            print(f"Error opening {filepath}: {e}")
            continue

        # dynamic sheets
        for sheet in xls.sheet_names:
            if "Hz" in sheet:
                freq_match = re.search(r"([0-9eE\+\-\.]+)Hz", sheet)
                if not freq_match:
                    continue
                freq = float(freq_match.group(1))
                df = pd.read_excel(xls, sheet_name=sheet)
                if "r_T (mol/cm²·s)" not in df.columns or "Time (s)" not in df.columns:
                    continue
                
                # Only include data after oscillations start (e.g., after 1 s)
                t_switching = 1.0  # seconds, adjust if needed
                mask = df["Time (s)"] >= t_switching
                if not mask.any():
                    continue  # skip if no data beyond switching time
                
                avg_rT = df.loc[mask, "r_T (mol/cm²·s)"].mean()
                
                all_freqs.append(freq)
                all_avg_rT.append(avg_rT)

                # ---- static reference (capture first valid one only) ----
        if static_rT is None and "Static_Summary" in xls.sheet_names:
            df_static = pd.read_excel(xls, sheet_name="Static_Summary")
            if "Average r_T (mol/cm²·s)" in df_static.columns:
                static_rT = df_static["Average r_T (mol/cm²·s)"].max()
                print(f"✅ Found static ⟨r_T⟩ = {static_rT:.3e} from {m['file']}")


    if not all_freqs:
        print("No dynamic data found.")
        return

    plot_bar_comparison(all_freqs, all_avg_rT, static_rT,
                        target_V, target_dGmin, target_dGmax, folder)


if __name__ == "__main__":
    main()
