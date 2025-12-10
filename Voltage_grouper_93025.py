import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.size": 20,        # increase overall font size
    "axes.labelsize": 20,   # bigger axis labels
    "axes.labelweight": "bold",
    "axes.titlesize": 24,   # bigger title
    "axes.titleweight": "bold",
    "legend.fontsize": 16,  # legend text size
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# ---------- helpers ----------
def list_excel_files(folder="."):
    return [f for f in os.listdir(folder) if f.lower().endswith(".xlsx")]

def parse_metadata_from_filename(fname):
    meta = {"file": fname}

    # Voltage
    v_match = re.search(r"_V_(-?\d+\.\d+)V", fname)
    if v_match:
        meta["V"] = float(v_match.group(1))

    # Binding energy
    dg_match = re.search(r"_dG_([0-9\.]+)-([0-9\.]+)eV", fname)
    if dg_match:
        meta["dGmin"] = float(dg_match.group(1))
        meta["dGmax"] = float(dg_match.group(2))

    # Frequencies (possibly multiple, separated by '-')
    f_match = re.search(r"freq_([0-9eE\+\-\.]+(?:-[0-9eE\+\-\.]+)*)", fname)
    if f_match:
        freq_str = f_match.group(1)
        try:
            freqs = [float(f) for f in freq_str.split("-")]
            meta["freqs"] = freqs
        except ValueError:
            meta["freqs"] = []
    else:
        meta["freqs"] = []

    return meta

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


label_map = {
    "t": "Time (s)",
    "curr": r"Current (mA/cm$^2$)",
    "r_T": r"$\mathbf{r_T}$ (mol/cm$^2\cdot$s)",
    "r_V": r"$\mathbf{r_V}$ (mol/cm$^2\cdot$s)",
    "thetaH": r"$\mathbf{\theta_{\mathrm{H}}}$",
    "GHad_eV": r"$\mathbf{\Delta G_{\mathrm{H}}}$ (eV)"
}

# ---------- main ----------
def main():
    folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
    files = list_excel_files(folder)
    if not files:
        print("No .xlsx files found.")
        return

    # Collect metadata
    metadata = [parse_metadata_from_filename(f) for f in files]

    # Menu 1: data type
    data_types = {"1": "curr", "2": "r_V", "3": "r_T", "4": "thetaH"}
    print("\nSelect data type to plot:")
    for k, v in data_types.items():
        print(f"{k}: {v}")
    choice = input("Enter choice #: ").strip()
    if choice not in data_types:
        print("Invalid choice.")
        return
    y_key = data_types[choice]

    # Menu 2: frequency
    freqs = sorted({f for m in metadata for f in m["freqs"]}, reverse=True)
    print("\nAvailable frequencies:")
    for i, f in enumerate(freqs, 1):
        print(f"{i}: {f} Hz")
    k = int(input("Pick frequency #: ").strip())
    target_freq = freqs[k - 1]

    # Map frequency to sheet index (assuming order: 2e4 → 2e3 → 2e2 → 2e1)
    freq_to_index = {1e4: 0, 1e3: 1, 1e2: 2, 1e1: 3}
    if target_freq not in freq_to_index:
        raise ValueError(f"No sheet index mapping for frequency {target_freq}")
    sheet_index = freq_to_index[target_freq]

    # Menu 3: dG ranges
    dg_ranges = sorted(set((m["dGmin"], m["dGmax"]) for m in metadata if target_freq in m["freqs"]))
    print("\nAvailable binding energies:")
    for i, (lo, hi) in enumerate(dg_ranges, 1):
        print(f"{i}: {lo:.2f} – {hi:.2f} eV")
    k = int(input("Pick dG range #: ").strip())
    dGmin, dGmax = dg_ranges[k - 1]

    # Menu 4: number of cycles
    n_cycles_to_show = int(input("\nHow many cycles do you want to plot? ").strip())

    # Filter metadata group
    group = [m for m in metadata if target_freq in m["freqs"]
             and abs(m["dGmin"] - dGmin) < 1e-6 and abs(m["dGmax"] - dGmax) < 1e-6]
    
    # Menu 5: voltages
    voltages = sorted({m["V"] for m in group})
    print("\nAvailable voltages:")
    for i, V in enumerate(voltages, 1):
        print(f"{i}: {V} V")
    
    choices = input("Enter voltage numbers (comma-separated, or 'all'): ").strip().lower()
    if choices == "all":
        selected_voltages = voltages
    else:
        try:
            idxs = [int(c.strip()) - 1 for c in choices.split(",")]
            selected_voltages = [voltages[i] for i in idxs if 0 <= i < len(voltages)]
        except ValueError:
            print("Invalid voltage selection, defaulting to all.")
            selected_voltages = voltages


    if not group:
        print("No matching files found.")
        return

    # Calculate period
    period = 1.0 / target_freq

    # Plot all voltages
    fig, ax = plt.subplots(figsize=(10, 8))
    for m in sorted(group, key=lambda x: x["V"]):
        if m["V"] not in selected_voltages:
            continue
        xls = pd.ExcelFile(os.path.join(folder, m["file"]))
        if sheet_index >= len(xls.sheet_names):
            raise ValueError(f"File {m['file']} does not have enough sheets for {target_freq} Hz")
        sheet = xls.sheet_names[sheet_index]

        data = read_sheet_columns_0_to_5(xls, sheet)

        # take the LAST n_cycles_to_show cycles
        t_max = data["t"].max()
        t_min = t_max - n_cycles_to_show * period
        mask = (data["t"] >= t_min) & (data["t"] <= t_max)

        t = data["t"][mask]
        y = data[y_key][mask]

        # downsample if too many points
        max_points = 2000
        if len(t) > max_points:
            step = len(t) // max_points
            t = t[::step]
            y = y[::step]

        # main trace
        line, = ax.plot(t, y, linewidth=4, alpha=0.9)
        ax.set_ylabel(label_map[y_key])
        mid_idx = len(t) // 2
        
        # find the max y-value of the current set of traces to detect the "top" trace
        ymax_current = y[mid_idx]
        
        # get global maximum mid-trace y across all selected voltages
        all_mid_vals = []
        for mm in sorted(group, key=lambda x: x["V"]):
            if mm["V"] in selected_voltages:
                xls_tmp = pd.ExcelFile(os.path.join(folder, mm["file"]))
                sheet_tmp = xls_tmp.sheet_names[sheet_index]
                data_tmp = read_sheet_columns_0_to_5(xls_tmp, sheet_tmp)
                mask_tmp = (data_tmp["t"] >= t_min) & (data_tmp["t"] <= t_max)
                all_mid_vals.append(data_tmp[y_key][mask_tmp][len(data_tmp[y_key][mask_tmp]) // 2])
        
        # get global y-axis range (based on all plotted traces)
        all_y_vals = []
        for mm in group:
            if mm["V"] in selected_voltages:
                xls_tmp = pd.ExcelFile(os.path.join(folder, mm["file"]))
                sheet_tmp = xls_tmp.sheet_names[sheet_index]
                data_tmp = read_sheet_columns_0_to_5(xls_tmp, sheet_tmp)
                mask_tmp = (data_tmp["t"] >= t_min) & (data_tmp["t"] <= t_max)
                all_y_vals.extend(data_tmp[y_key][mask_tmp])
        
        global_y_range = max(all_y_vals) - min(all_y_vals)
        global_max_mid = max(all_mid_vals)
        
        # offset is now consistent across all traces
        if np.isclose(ymax_current, global_max_mid):
            offset = -0.05 * global_y_range   # top trace → put label below
        else:
            offset =  0.04 * global_y_range   # others → put label above
        
        
        # add inline label
        ax.text(
            t[mid_idx], y[mid_idx] + offset,
            f"{m['V']} V",
            fontsize=20, weight='bold',
            color=line.get_color(),
            va='center', ha='left',
            clip_on = True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2")
        )
    ax2 = ax.twinx()
    ghad_line, = ax2.plot(
        t, data["GHad_eV"][mask],
        color="gray", linestyle="--", linewidth=4, alpha=0.7,
        zorder=1  # keep this in the background
    )
    ax2.set_ylabel("Binding Energy (eV)")
    ax2.tick_params(axis="y", labelcolor="gray")
    
    # inline label for ΔG
    ax2.text(
        t[mid_idx], data["GHad_eV"][mask][mid_idx] + 0.05,
        r"$\boldsymbol{\Delta G}$",
        fontsize=20, weight='bold', color="gray",
        va='center', ha='left',
        clip_on = True,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2")
    )
    
    # add a label to the gray line
    ghad_line.set_label(r"$\boldsymbol{\Delta G}$")
    
    plt.subplots_adjust(left=0.15)
    ax.xaxis.set_major_locator(MultipleLocator(0.02)) 

    ax.set_title(
        f"{label_map[y_key]} vs Time (with $\\boldsymbol{{\\Delta G}}$)\n")

    # Save and show
    outname = os.path.join(folder,
        f"{y_key}_vs_time_{n_cycles_to_show}cycles_freq{target_freq}_dG{dGmin}-{dGmax}.png")
    fig.savefig(outname, dpi=300)
    print("Saved", outname)
    plt.show()


if __name__ == "__main__":
    main()
