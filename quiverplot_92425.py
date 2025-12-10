import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,        # increase overall font size
    "axes.labelsize": 14,   # bigger axis labels
    "axes.labelweight": "bold",
    "axes.titlesize": 18,   # bigger title
    "axes.titleweight": "bold",
    "legend.fontsize": 12,  # legend text size
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

save_folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\plots"

# ---------- helpers ----------
def list_excel_files(folder="."):
    return [f for f in os.listdir(folder) if f.lower().endswith(".xlsx")]

def choose_excel_file(folder="."):
    files = list_excel_files(folder)
    if not files:
        print("No .xlsx files found.")
        return None
    print("\nAvailable Excel files:")
    for i, f in enumerate(files, 1):
        print(f"{i}: {f}")
    while True:
        try:
            k = int(input("Pick file #: ").strip())
            if 1 <= k <= len(files):
                return os.path.join(folder, files[k-1])
        except ValueError:
            pass
        print("Invalid choice.")

def parse_freq_from_sheet(sheet_name):
    s = str(sheet_name).replace("Hz", "")
    try:
        return float(s)
    except:
        m = re.search(r"([0-9eE+\-\.]+)", s)
        return float(m.group(1)) if m else None

def read_sheet_columns_0_to_5(xls, sheet_name):
    # Expect columns:
    # 0 Time(s), 1 GHad(eV), 2 Current, 3 r_T, 4 r_V, 5 thetaH
    df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
    arr = df.to_numpy()
    if arr.shape[1] < 6:
        raise ValueError(f"Sheet '{sheet_name}' has {arr.shape[1]} columns; need >= 6")
    return {
        "t":       arr[:, 0].astype(float),
        "GHad_eV": arr[:, 1].astype(float),
        "curr":    arr[:, 2].astype(float),
        "r_T":     arr[:, 3].astype(float),
        "r_V":     arr[:, 4].astype(float),
        "thetaH":  arr[:, 5].astype(float),
    }

# ---------- plotting ----------
def quiver_panel_magnitude(ax, g_label, rows, colors, values, x_max, labels_above=False):
    n = len(rows)
    ys = np.arange(n)[::-1]  # top→bottom rows

    # zero line
    ax.axvline(0, color="0.85", lw=1)

    for y, val, c in zip(ys, values, colors):
        # draw arrow from 0 to val
        ax.quiver(0, y, val, 0.0,
                  angles="xy", scale_units="xy", scale=1,
                  color=c, width=0.01, headlength=6, headwidth=4, zorder=3)

        if labels_above:
            # label at arrow MIDPOINT, slightly above it
            if np.isclose(val, 0.0):
                x_text = 0.02 * x_max    # nudge off the zero line
            else:
                x_text = 0.5 * val       # midpoint
                x_text = np.clip(x_text, -0.95 * x_max, 0.95 * x_max)

            y_text = y + 0.15            # vertical offset above the arrow
            ax.text(x_text, y_text, f"{abs(val):.2e}",
                    ha="center", va="bottom", fontsize=10, color=c,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
                    zorder=4, fontweight='bold')
        else:
            # fallback: put label on right edge
            ax.text(0.98 * x_max, y, f"{abs(val):.2e}",
                    ha="right", va="center", fontsize=10, color=c,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
                    zorder=4, fontweight='bold')

    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(rows, fontweight='bold')
    ax.set_title(g_label, fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)




def main():
    # ---- pick file & sheet ----
    folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
    filepath = choose_excel_file(folder)
    if not filepath:
        return
    # --- right after filepath is chosen ---
    fname = os.path.basename(filepath)
    voltage_match = re.search(r"_V_(-?\d+\.\d+)V", fname)
    if voltage_match:
        voltage_str = voltage_match.group(1) + " V"
    else:
        voltage_str = "N/A"
    xls = pd.ExcelFile(filepath)
    sheets = xls.sheet_names
    print("\nSheets:")
    for i, s in enumerate(sheets, 1):
        print(f"{i}: {s}")
    while True:
        try:
            k = int(input("Pick sheet #: ").strip())
            if 1 <= k <= len(sheets):
                sheet = sheets[k-1]
                break
        except ValueError:
            pass
        print("Invalid choice.")

    data = read_sheet_columns_0_to_5(xls, sheet)
    freq = parse_freq_from_sheet(sheet) or 1.0

    # ---- skip first full cycle ----
    period = 2.0 / freq
    t_switching = 0.5
    mask = data["t"] >= (t_switching + period)

    G = data["GHad_eV"][mask]
    rV = data["r_V"][mask]
    rT = data["r_T"][mask]
    I  = data["curr"][mask]

    # ---- average by GHad ----
    unique_G = np.unique(np.round(G, 10))
    # keep sorted order
    unique_G = np.sort(unique_G)

    row_vals_rates = []  # per GHad: (avg_rV, avg_rT)
    row_vals_curr  = []  # per GHad: (avg_I,)
    for g in unique_G:
        m = np.isclose(G, g, rtol=1e-8, atol=1e-12)
        row_vals_rates.append( (float(np.mean(rV[m])), float(np.mean(rT[m]))) )
        row_vals_curr.append(  (float(np.mean(I[m])),) )

    # ---- scaling for magnitude axis ----
    # Fig 1 (rates) uses max of |rV| and |rT| across all GHad
    all_rates = np.array(row_vals_rates).ravel()
    max_rate_mag = np.max(np.abs(all_rates)) if all_rates.size else 1.0
    x_max_rates = 1.3 * max(1e-30, max_rate_mag)

    # Fig 2 (current) uses max of |I| across all GHads
    all_I = np.array(row_vals_curr).ravel()
    max_I_mag = np.max(np.abs(all_I)) if all_I.size else 1.0
    x_max_curr = 1.3 * max(1e-30, max_I_mag)

    # ---- Figure 1: r_V & r_T, columns = GHad ----
    nG = len(unique_G)
    fig1, axes1 = plt.subplots(1, nG, figsize=(4.2*nG, 4.0), sharey=True, squeeze=False)
    axes1 = axes1[0]
    for j, g in enumerate(unique_G):
        ax = axes1[j]
        g_label = f"GHad = {g:.2f} eV"
        (v_rV, v_rT) = row_vals_rates[j]
        quiver_panel_magnitude(
            ax,
            g_label=g_label,
            rows=["r_V (mol/cm²·s)", "r_T (mol/cm²·s)"],
            colors=["tab:orange", "tab:red"],
            values=[v_rV, v_rT],
            x_max=x_max_rates,
            labels_above = True
        )
        if j == 0:
            ax.set_ylabel("", fontweight='bold')  # y labels already set per panel
        ax.set_xlabel("Magnitude (→ pos, ← neg)", fontweight='bold', labelpad=15)
    fig1.suptitle(f"Volmer and Tafel Rates\nV = {voltage_str}, (f={freq:.0f} Hz)", fontsize=12, fontweight='bold')
    fig1.tight_layout()
    outname1 = os.path.join(save_folder, f"rates_GHad_{os.path.splitext(fname)[0]}_{sheet}.png")
    fig1.savefig(outname1, dpi=300, bbox_inches="tight")
    print("Saved", outname1)
    plt.show()

    # ---- Figure 2: Current only, columns = GHad ----
    fig2, axes2 = plt.subplots(1, nG, figsize=(3.8*nG, 3.2), sharey=True, squeeze=False)
    axes2 = axes2[0]
    for j, g in enumerate(unique_G):
        ax = axes2[j]
        g_label = f"GHad = {g:.2f} eV"
        (v_I,) = row_vals_curr[j]
        quiver_panel_magnitude(
            ax,

            g_label=g_label,
            rows=["Current (mA/cm²)"],
            colors=["tab:blue"],
            values=[v_I],
            x_max=x_max_curr,
            labels_above = True
        )
        ax.set_xlabel("Magnitude (→ pos, ← neg)")
    fig2.suptitle(f"Current per GHad (averaged over time)\nV = {voltage_str}, (f={freq:.2e} Hz)", fontsize=12)
    fig2.tight_layout()
    # save
    outname2 = os.path.join(save_folder, f"current_GHad_{os.path.splitext(fname)[0]}_{sheet}.png")
    fig2.savefig(outname2, dpi=300, bbox_inches="tight")
    print("Saved", outname2)
    plt.show()

if __name__ == "__main__":
    main()
