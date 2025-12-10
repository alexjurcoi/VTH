import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")

FREQ_COLORS = {
    20: "tab:pink",       # pink
    200: "tab:green",     # green
    2000: "tab:orange",   # orange
    20000: "red"          # red
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


# ---------- bar chart ----------
def plot_bar_comparison(all_freqs, all_avg_rT, static_rT, target_V, target_dGmin, target_dGmax, folder):
    all_freqs, all_avg_rT = zip(*sorted(zip(all_freqs, all_avg_rT), key=lambda x: x[0]))
    n = len(all_freqs)

    # --- prepare grouped bars ---
    x = np.arange(n)
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [FREQ_COLORS.get(int(round(f)), "#b22222") for f in all_freqs]
    bars_dyn = sns.barplot(x - bar_width/2, all_avg_rT, width=bar_width,
                      color=colors, edgecolor="black", label="Dynamic ⟨rₜ⟩")
    bars_stat = sns.barplot(x + bar_width/2, [static_rT]*n, width=bar_width,
                       color="grey", edgecolor="black", label="Static ⟨rₜ⟩")

    # --- annotate each bar ---
    for bar in bars_dyn:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height*1.01, f"{height:.2e}",
                ha="center", va="bottom", fontsize=10)
    for bar in bars_stat:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height*1.01, f"{height:.2e}",
                ha="center", va="bottom", fontsize=10, color="gray")

    # --- x-axis frequency labels ---
    freq_labels = [f"$10^{{{int(np.log10(f))}}}$" if np.log10(f).is_integer() else f"{f:.0e}" for f in all_freqs]
    ax.set_xticks(x)
    ax.set_xticklabels(freq_labels)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average $r_T$ (mol cm$^{-2}$ s$^{-1}$)")

    # --- title and style ---
    ax.set_title(f"Average $r_T$ Comparison by Frequency\n(V = {target_V:.2f} V, ΔG = {target_dGmin:.2f}–{target_dGmax:.2f} eV)")
    legend = ax.legend(
    frameon=True,
    loc="bottom right",
    facecolor="white",      # solid white background
    edgecolor="black",      # black border
    framealpha=1.0,          # fully opaque
)
    # --- add y-axis headroom (prevents bars from touching top) ---
    ymax = max(max(all_avg_rT), static_rT) * 1.10   # 10% headroom
    ax.set_ylim(0, ymax)

    plt.tight_layout()
    outname = os.path.join(
        folder,
        f"avg_rT_bar_vs_freq_V{target_V:.2f}_dG{target_dGmin:.2f}-{target_dGmax:.2f}.png"
    )
    plt.savefig(outname, dpi=400, bbox_inches="tight")
    print(f"✅ Saved bar chart → {outname}")
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

        # static reference
        if "Static_Summary" in xls.sheet_names:
            df_static = pd.read_excel(xls, sheet_name="Static_Summary")
            if "Average r_T (mol/cm²·s)" in df_static.columns:
                static_rT = df_static["Average r_T (mol/cm²·s)"].max()

    if not all_freqs:
        print("No dynamic data found.")
        return

    plot_bar_comparison(all_freqs, all_avg_rT, static_rT,
                        target_V, target_dGmin, target_dGmax, folder)


if __name__ == "__main__":
    main()
