import os
import re
import math
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sys


# ---------- parsing helper ----------
def parse_filename(filename):
    """
    Extracts parameters (freq, V, dG, type) from filename.
    """
    params = {}
    params["type"] = "rates" if filename.startswith("rates") else "current"

    # Frequency: take the number before "Hz" at the END
    freq_match = re.search(r'_([0-9eE\+\-\.]+)Hz(?:\.png)?$', filename)
    if freq_match:
        params['freq'] = freq_match.group(1)

    # Voltage (look for _V_-0.40V_)
    v_match = re.search(r'_V_(-?\d+\.\d+)V', filename)
    if v_match:
        params['V'] = float(v_match.group(1))

    # Binding energy (dG, e.g. _dG_0.05-0.15eV)
    dg_match = re.search(r'_dG_([0-9eE\+\-\.]+(?:-[0-9eE\+\-\.]+)?)eV', filename)
    if dg_match:
        params['dG'] = dg_match.group(1)

    return params


# ---------- grid builder ----------
def make_grids(folder, group_by, group_value,
               data_type="current", nrows=2, ncols=2,
               outprefix="grid", target_px=1200):
    """
    Make multiple grids (nrows x ncols) of images grouped by one parameter.
    """
    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    selected = []

    for f in files:
        p = parse_filename(f)
        if p["type"] == data_type and str(p.get(group_by)) == str(group_value):
            selected.append((f, p))

    if not selected:
        print(f"No {data_type} files found for {group_by}={group_value}")
        return

    # Sort order
    def sort_key(item):
        p = item[1]
        if group_by != "V" and "V" in p:
            return p["V"]
        elif group_by != "dG" and "dG" in p:
            try:
                return float(p["dG"])
            except:
                return 0
        elif group_by != "freq" and "freq" in p:
            return float(p["freq"])
        return 0

    selected.sort(key=sort_key)

    # Page through results in chunks of grid size
    per_page = nrows * ncols
    total_pages = math.ceil(len(selected) / per_page)

    for page in range(total_pages):
        subset = selected[page * per_page : (page + 1) * per_page]

        figsize = (ncols * target_px / 100, nrows * target_px / 100)  # inches @ dpi=100
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=100)
        axes = axes.flatten()

        for ax, (fname, params) in zip(axes, subset):
            img = imread(os.path.join(folder, fname))
            ax.imshow(img)
            ax.axis("off")

            label = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ["type", group_by])
            ax.set_title(label, fontsize=8)

        for ax in axes[len(subset):]:
            ax.axis("off")

        plt.tight_layout()
        outname2 = f"{outprefix}_{data_type}_{group_by}{group_value}_page{page+1}.png"
        location = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\plots\Grouped Plots"
        outname = os.path.join(location, outname2)
        plt.savefig(outname, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print(f"Saved {outname} ({len(subset)} images)")


# ---------- interactive menus ----------
if __name__ == "__main__":
    folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\plots"

    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    parsed = [parse_filename(f) for f in files]

    # Menu 1: data type
    data_types = sorted(set(p["type"] for p in parsed))
    print("\nSelect data type:")
    for i, dt in enumerate(data_types, 1):
        print(f"{i}: {dt}")
    choice = int(input("Enter choice #: "))
    data_type = data_types[choice - 1]

    # Menu 2: group_by
    group_keys = ["freq", "V", "dG"]
    print("\nGroup by:")
    for i, gk in enumerate(group_keys, 1):
        print(f"{i}: {gk}")
    choice = int(input("Enter choice #: "))
    group_by = group_keys[choice - 1]

    # Menu 3: group_value
    # Gather all values for the chosen group_by, only for selected data_type
    values = [p[group_by] for p in parsed if group_by in p and p["type"] == data_type]
    
    if not values:
        print(f"No values found for {group_by} with type {data_type}")
        sys.exit()
    
    # Remove duplicates and sort nicely
    def safe_float(x):
        try:
            return float(x)
        except:
            return x
    
    unique_values = sorted(set(values), key=safe_float)
    
    print(f"\nAvailable values for {group_by}:")
    for i, val in enumerate(unique_values, 1):
        print(f"{i}: {val}")
    choice = int(input("Enter choice #: "))
    group_value = unique_values[choice - 1]
    
    # Cast voltage to float so filtering works
    if group_by == "V":
        group_value = float(group_value)

    # Run grid builder
    make_grids(folder, group_by=group_by, group_value=group_value,
               data_type=data_type, outprefix="grid")
