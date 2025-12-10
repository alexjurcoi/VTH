import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Frequencies of interest
FREQ_ORDER = [1e-4, 1e0, 1e1, 1e2]

MECH_COLORS = {
    "Static": "gray",
    "VT": "#1f77b4",   # blue
    "VH": "#d62728",   # red
}

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
})

def list_excel_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".xlsx")]

# ---------- metadata parsing ----------
def parse_meta(fname):
    meta = {"file": fname}

    # Voltage
    v = re.search(r"_V_(-?\d+\.\d+)", fname)
    if v:
        meta["V"] = float(v.group(1))

    # ΔG window
    dg = re.search(r"_+dG_([\-0-9\.]+)-([\-0-9\.]+)eV", fname)
    if dg:
        meta["dGmin"] = float(dg.group(1))
        meta["dGmax"] = float(dg.group(2))

    # Mechanism via kT
    kT_match = re.search(r"kT=([0-9eE\+\-\.]+)", fname)
    if kT_match:
        kT_val = float(kT_match.group(1))
        meta["is_VH"] = (kT_val == 0.0)
        meta["is_VT"] = (kT_val > 0.0)
    else:
        meta["is_VH"] = False
        meta["is_VT"] = False

    return meta

# ---------- dynamic extraction ----------
def extract_dynamic_values(xls):
    """
    Returns dict: {frequency: [avg_rT_values]} for the four target freqs.
    """
    results = {}

    for sheet in xls.sheet_names:
        if not sheet.endswith("Hz"):
            continue

        try:
            freq = float(sheet.replace("Hz", ""))
        except ValueError:
            continue

        if freq not in FREQ_ORDER:
            continue

        df = pd.read_excel(xls, sheet_name=sheet)

        if "r_T (mol/cm²·s)" not in df.columns:
            continue

        # skip first row (switching point), use rest
        if len(df) <= 1:
            continue

        rT_mean = df.loc[df.index >= 1, "r_T (mol/cm²·s)"].mean()
        results.setdefault(freq, []).append(rT_mean)

    return results

# ---------- static extraction ----------
def extract_static(xls):
    if "Static_Summary" not in xls.sheet_names:
        return None

    df = pd.read_excel(xls, sheet_name="Static_Summary")
    col = "Average r_T (mol/cm²·s)"
    if col in df.columns:
        return df[col].max()
    return None


def main():
    folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"

    files = list_excel_files(folder)
    metadata = [parse_meta(f) for f in files]

    # --- Voltage selection ---
    voltages = sorted({m["V"] for m in metadata})
    print("\nAvailable voltages:")
    for i, V in enumerate(voltages, 1):
        print(f"{i}: {V} V")
    target_V = voltages[int(input("Pick voltage #: ")) - 1]

    # --- ΔG window selection ---
    dg_windows = sorted({(m["dGmin"], m["dGmax"]) for m in metadata})
    print("\nAvailable ΔG windows:")
    for i, (a, b) in enumerate(dg_windows, 1):
        print(f"{i}: {a} – {b} eV")
    target_dGmin, target_dGmax = dg_windows[int(input("Pick ΔG window #: ")) - 1]

    # --- Split VT vs VH files for this V, ΔG ---
    VT_files, VH_files = [], []
    for m in metadata:
        if np.isclose(m["V"], target_V) and np.isclose(m["dGmin"], target_dGmin) and np.isclose(m["dGmax"], target_dGmax):
            if m["is_VT"]:
                VT_files.append(m["file"])
            elif m["is_VH"]:
                VH_files.append(m["file"])

    print("\nVT files for this selection:", VT_files)
    print("VH files for this selection:", VH_files)

    if not VT_files and not VH_files:
        print("No VT or VH files found for this V and ΔG window.")
        return

    # --- Static value (from any matching file that has it) ---
    static_rT = None
    for f in VT_files + VH_files:
        xls = pd.ExcelFile(os.path.join(folder, f))
        static_rT = extract_static(xls)
        if static_rT is not None:
            break

    if static_rT is None:
        print("WARNING: No Static_Summary found; static r_T set to NaN.")
        static_rT = np.nan

    # --- VT dynamic averages ---
    VT_values = {freq: [] for freq in FREQ_ORDER}
    for fname in VT_files:
        xls = pd.ExcelFile(os.path.join(folder, fname))
        vals = extract_dynamic_values(xls)
        for freq, arr in vals.items():
            VT_values[freq].extend(arr)

    VT_avg = {f: (np.mean(VT_values[f]) if VT_values[f] else np.nan)
              for f in FREQ_ORDER}

    # --- VH dynamic averages ---
    VH_values = {freq: [] for freq in FREQ_ORDER}
    for fname in VH_files:
        xls = pd.ExcelFile(os.path.join(folder, fname))
        vals = extract_dynamic_values(xls)
        for freq, arr in vals.items():
            VH_values[freq].extend(arr)

    VH_avg = {f: (np.mean(VH_values[f]) if VH_values[f] else np.nan)
              for f in FREQ_ORDER}

    print("\nVT_avg:", VT_avg)
    print("VH_avg:", VH_avg)
    print("Static r_T:", static_rT)

    # Decide which mechanisms actually have data (non-NaN)
    mechanisms = []
    if not np.isnan(static_rT):
        mechanisms.append("Static")
    if any(not np.isnan(VT_avg[f]) for f in FREQ_ORDER):
        mechanisms.append("VT")
    if any(not np.isnan(VH_avg[f]) for f in FREQ_ORDER):
        mechanisms.append("VH")

    if "VH" not in mechanisms:
        print("NOTE: No VH data for this V and ΔG window; VH bars will be omitted.")

    # --- Grouped bar plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(FREQ_ORDER))

    # width based on how many mechanisms we actually plot
    n_mech = len(mechanisms)
    width = 0.8 / max(n_mech, 1)
    offsets = np.linspace(-0.4, 0.4, n_mech)[:n_mech]

    for i, mech in enumerate(mechanisms):
        if mech == "Static":
            heights = [static_rT] * len(FREQ_ORDER)
        elif mech == "VT":
            heights = [VT_avg[f] for f in FREQ_ORDER]
        else:  # VH
            heights = [VH_avg[f] for f in FREQ_ORDER]

        ax.bar(x + offsets[i], heights, width,
               label=mech, color=MECH_COLORS[mech])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.0e}" for f in FREQ_ORDER])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average r_T (mol/cm²·s)")
    ax.set_title(f"Static vs VT vs VH\nV = {target_V}, ΔG = {target_dGmin}-{target_dGmax}")

    # symlog to handle negative/positive rates
    ax.set_yscale("symlog", linthresh=1e-12)

    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
