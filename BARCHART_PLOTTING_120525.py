import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# --- The four frequencies you want ---
FREQ_ORDER = [1e-4, 1e0, 1e1, 1e2]

MECH_COLORS = {
    "Static": "gray",
    "VT": "#1f77b4",    # blue
    "VH": "#d62728",    # red
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

def parse_meta(fname):
    meta = {"file": fname}

    # voltage
    v = re.search(r"_V_(-?\d+\.\d+)", fname)
    if v:
        meta["V"] = float(v.group(1))
    
    # ΔG window
    dg = re.search(r"_dG_([0-9\.]+)-([0-9\.]+)eV", fname)
    if dg:
        meta["dGmin"] = float(dg.group(1))
        meta["dGmax"] = float(dg.group(2))

    # Mechanism based on kT numeric value
    # Looks for kT=... and determines if it is effectively zero or positive
    kT = re.search(r"kT=([0-9eE\+\-\.]+)", fname)
    if kT:
        kT_val = float(kT.group(1))
        meta["is_VT"] = kT_val > 0
        meta["is_VH"] = kT_val == 0 
    else:
        meta["is_VT"] = False
        meta["is_VH"] = False

    return meta


def extract_dynamic_values(xls):
    """
    Extract steady-state average r_T values for each frequency sheet.
    This version is robust to:
    - Unicode issues in superscripts
    - Slightly different header formatting
    - Extra spaces
    - Different sheet name formatting
    """
    result = {}

    # List all sheets for debugging
    # print("Sheets in file:", xls.sheet_names)

    for sheet in xls.sheet_names:

        # Accept sheets that contain 'Hz' anywhere
        if "Hz" not in sheet:
            continue

        # Try to extract frequency numerically
        freq_str = sheet.replace("Hz", "").strip()
        try:
            freq = float(freq_str)
        except:
            # print("Could not parse freq for sheet:", sheet)
            continue

        # Only include desired frequencies
        if freq not in FREQ_ORDER:
            continue

        # Load the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet)

        # --- Robust header detection ---
        # Time column
        time_col = next(
            (c for c in df.columns if "time" in c.lower()),
            None
        )

        # r_T column
        rT_col = next(
            (c for c in df.columns if c.strip().startswith("r_T")),
            None
        )

        # Diagnostics if missing
        if time_col is None or rT_col is None:
            print(f"\n[WARN] Skipping sheet '{sheet}' (missing columns)")
            print("Columns found:", list(df.columns))
            continue

        # --- Compute steady-state mean ---
        # Use last second of simulation OR last 20% if shorter
        tmax = df[time_col].max()
        threshold = max(tmax - 1.0, tmax * 0.8)

        mask = df[time_col] >= threshold
        if mask.sum() == 0:
            print(f"[WARN] No steady-state points in sheet '{sheet}'")
            continue

        avg_rT = df.loc[mask, rT_col].mean()

        result.setdefault(freq, []).append(avg_rT)

    return result


def extract_static(xls):
    for sheet in xls.sheet_names:
        # FIX: Remove underscores AND spaces to match "Static_Summary"
        norm_name = sheet.replace(" ", "").replace("_", "").lower()
        
        if "staticsummary" in norm_name:
            df = pd.read_excel(xls, sheet_name=sheet)
            if "Average r_T (mol/cm²·s)" in df:
                return df["Average r_T (mol/cm²·s)"].max()
    return None


def main():
    # UPDATE THIS PATH TO YOUR ACTUAL FOLDER
    folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
    
    if not os.path.exists(folder):
        print(f"Error: Folder not found: {folder}")
        return

    files = list_excel_files(folder)
    if not files:
        print("No .xlsx files found in folder.")
        return

    metadata = [parse_meta(f) for f in files]

    # Let user pick voltage
    voltages = sorted({m.get("V") for m in metadata if "V" in m})
    if not voltages:
        print("Could not parse voltages from filenames.")
        return

    print("\nAvailable voltages:")
    for i, V in enumerate(voltages, 1):
        print(f"{i}: {V}  V")
    try:
        k = int(input("Pick voltage #: "))
        target_V = voltages[k-1]
    except (IndexError, ValueError):
        print("Invalid selection.")
        return

    # Let user pick ΔG window
    dg_windows = sorted({(m.get("dGmin"), m.get("dGmax")) for m in metadata if "dGmin" in m})
    print("\nAvailable ΔG windows:")
    for i, (a, b) in enumerate(dg_windows, 1):
        print(f"{i}: {a} – {b}")
    try:
        k = int(input("Pick ΔG window #: "))
        target_dGmin, target_dGmax = dg_windows[k-1]
    except (IndexError, ValueError):
        print("Invalid selection.")
        return

    # Split into VT and VH files
    VT_files = []
    VH_files = []
    
    print("\n--- MECHANISM PARSE CHECK ---")
    for m in metadata:
        print(m["file"], "  is_VT=", m["is_VT"], "  is_VH=", m["is_VH"], 
              "  V=", m.get("V"), " dG=", m.get("dGmin"), m.get("dGmax"))

    
    for m in metadata:
        # Match user selection
        if (np.isclose(m.get("V", -999), target_V) and 
            np.isclose(m.get("dGmin", -999), target_dGmin) and 
            np.isclose(m.get("dGmax", -999), target_dGmax)):
            
            if m["is_VT"]:
                VT_files.append(m["file"])
            elif m["is_VH"]:
                VH_files.append(m["file"])

    print(f"\nFound {len(VT_files)} VT files and {len(VH_files)} VH files.")
    
    # --- Extract STATIC value from any file in either group ---
    static_rT = None
    # Check all relevant files for the static summary
    for f in VT_files + VH_files:
        try:
            xls = pd.ExcelFile(os.path.join(folder, f))
            static_rT = extract_static(xls)
            if static_rT is not None:
                print(f"Static value found in: {f}")
                break
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # FIX: Handle case where Static Summary is missing
    if static_rT is None:
        print("WARNING: Could not find 'Static Summary' or 'r_T' in any file. Defaulting Static to 0.")
        static_rT = 0.0

    # --- Extract VT averages ---
    VT_values = {freq: [] for freq in FREQ_ORDER}
    for f in VT_files:
        xls = pd.ExcelFile(os.path.join(folder, f))
        vals = extract_dynamic_values(xls)
        for freq in vals:
            VT_values[freq].extend(vals[freq])

    VT_avg = {freq: np.mean(VT_values[freq]) if VT_values[freq] else 0.0 for freq in FREQ_ORDER}

    # --- Extract VH averages ---
    VH_values = {freq: [] for freq in FREQ_ORDER}
    for f in VH_files:
        xls = pd.ExcelFile(os.path.join(folder, f))
        vals = extract_dynamic_values(xls)
        for freq in vals:
            VH_values[freq].extend(vals[freq])

    VH_avg = {freq: np.mean(VH_values[freq]) if VH_values[freq] else 0.0 for freq in FREQ_ORDER}

    # --- Build grouped bar plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(FREQ_ORDER))
    width = 0.25

    # Static
    ax.bar(x - width, [static_rT]*len(FREQ_ORDER), width, label="Static", color=MECH_COLORS["Static"])

    # VT
    ax.bar(x, [VT_avg[f] for f in FREQ_ORDER], width, label="VT", color=MECH_COLORS["VT"])

    # VH
    ax.bar(x + width, [VH_avg[f] for f in FREQ_ORDER], width, label="VH", color=MECH_COLORS["VH"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.0e}" for f in FREQ_ORDER])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average rT (mol/cm²·s)")
    ax.set_title(f"Static vs VT vs VH\nV = {target_V}, ΔG = {target_dGmin}-{target_dGmax}")

    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()