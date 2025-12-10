import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

plt.rcParams.update({
    "font.size": 14,        # increase overall font size
    "axes.labelsize": 16,   # bigger axis labels
    "axes.labelweight": "bold",
    "axes.titlesize": 18,   # bigger title
    "axes.titleweight": "bold",
    "legend.fontsize": 12,  # legend text size
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 4
})

def simplify_filename(filename):
    """
    Simplifies a complex filename into a more readable string.
    """
    # Remove extension
    name = filename.replace(".xlsx", "")

    # Break into parts by "_"
    parts = name.split("_")

    # Dictionary to store values
    decoded = {}

    for i, p in enumerate(parts):
        if p.startswith("kV="):
            decoded["kV"] = p.split("=")[1]
        elif p.startswith("kT=") or p.startswith("kH="):
            decoded["kT/kH"] = p.split("=")[1]
        elif p == "freq":
            decoded["freq"] = parts[i+1]
        elif p == "beta":
            # Handle "__" as multiple betas
            beta_str = parts[i+1].replace("__", ",")
            decoded["β"] = beta_str
        elif p == "dG":
            decoded["dG"] = parts[i+1]
        elif p == "V":
            decoded["V"] = parts[i+1]

    # Build simplified string
    simple = []
    if "kV" in decoded:
        simple.append(f"kV={decoded['kV']}")
    if "kT/kH" in decoded:
        simple.append(f"kT/kH={decoded['kT/kH']}")
    if "freq" in decoded:
        simple.append(f"freq={decoded['freq']}")
    if "β" in decoded:
        simple.append(f"β={decoded['β']}")
    if "dG" in decoded:
        simple.append(f"dG={decoded['dG']}")
    if "V" in decoded:
        simple.append(f"V={decoded['V']}")

    return ", ".join(simple)



variable_map = {
    "1": ("t", "Time (s)"),
    "2": ("thetaH", "θ_H (fraction)"),
    "3": ("curr", "Current (mA/cm²)"),
    "4": ("r_T", "r_T (mol/cm²·s)"),
    "5": ("r_V", "r_V (mol/cm²·s)"),
    "6": ("GHad_eV", "GHad (eV)")
}

step_sizes = {
    2e4: 1.25e-6,
    2e3: 1.25e-5,
    2e2: 1.25e-4,
    2e1: 1.25e-3,
}

def list_excel_files(folder="."):
    """Return a list of Excel files in the folder."""
    return [f for f in os.listdir(folder) if f.endswith(".xlsx")]

def choose_excel_file(folder="."):
    """Interactive function to choose an Excel file from a folder."""
    files = list_excel_files(folder)
    if not files:
        print("No Excel files found in folder.")
        return None
    
    print("\nAvailable Excel files:")
    for i, f in enumerate(files, start=1):
        print(f"{i}: {simplify_filename(f)}")

    
    choice = input("Enter file number: ").strip()
    try:
        idx = int(choice) - 1
        return os.path.join(folder, files[idx])
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None

def load_dyn_results_from_excel(filepath):
    """
    Loads data from an Excel file into a list of dictionaries, mapping
    columns by their headers instead of index.
    """
    xls = pd.ExcelFile(filepath)
    dyn_results = []
    
    # Map the Excel column names to the variable names used in the script
    column_map = {
        'Time (s)': 't',
        'GHad (eV)': 'GHad_eV',
        'Current (mA/cm²)': 'curr',
        'r_T (mol/cm²·s)': 'r_T',
        'r_V (mol/cm²·s)': 'r_V',
        'θ_H': 'thetaH'
    }

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        
        # Rename columns to match the variable names in the script
        df.rename(columns=column_map, inplace=True)

        # Check for expected columns after renaming
        expected_cols = list(column_map.values())
        if not all(col in df.columns for col in expected_cols):
            print(f"Warning: Sheet '{sheet}' is missing one or more required columns after renaming.")
            print(f"Available columns: {df.columns.tolist()}")
            continue

        try:
            freq = float(sheet.replace("Hz", ""))
        except:
            freq = None
        
        res = {
            "freq": freq,
            "t": df['t'].to_numpy(),
            "GHad_eV": df['GHad_eV'].to_numpy(),
            "curr": df['curr'].to_numpy(),
            "r_T": df['r_T'].to_numpy(),
            "r_V": df['r_V'].to_numpy(),
            "thetaH": df['thetaH'].to_numpy()
        }
        dyn_results.append(res)
    
    return dyn_results

def interactive_plot_xy(dyn_results, save_folder="plots"):
    """
    Completely rewritten plotting function with better logic and cleaner implementation.
    """
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(save_folder, exist_ok=True)
    
    # Available variables with their keys and display names
    variables = {
        't': 'Time (s)',
        'GHad_eV': 'GHad (eV)', 
        'curr': 'Current (mA/cm²)',
        'r_T': '$r_T$ (mol/cm²·s)',
        'r_V': '$r_V$ (mol/cm²·s)',
        'thetaH': 'θ_H (fraction)'
    }
    
    print("\n" + "="*60)
    print("INTERACTIVE PLOTTING TOOL")
    print("="*60)
    
    # Step 1: Choose frequency
    print("\nStep 1: Choose Frequency")
    print("-" * 25)
    available_freqs = [res["freq"] for res in dyn_results]
    for i, freq in enumerate(available_freqs, 1):
        print(f"{i}: {freq:.2e} Hz")
    
    while True:
        try:
            freq_choice = int(input("\nEnter frequency number: ").strip())
            if 1 <= freq_choice <= len(available_freqs):
                selected_data = dyn_results[freq_choice - 1]
                selected_freq = available_freqs[freq_choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Step 2: Choose time range (cycles)
    print(f"\nStep 2: Choose Time Range")
    print("-" * 26)
    while True:
        try:
            cycles = float(input("Enter number of cycles to plot: ").strip())
            if cycles > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Calculate time mask
    period = 2.0 / selected_freq
    t_switching = 1
    t_start = t_switching
    t_end = t_start + cycles * period
    time_mask = (selected_data['t'] >= t_start) & (selected_data['t'] < t_end)
    
    
    # Step 3: Choose X-axis
    print(f"\nStep 3: Choose X-axis Variable")
    print("-" * 30)
    var_list = list(variables.keys())
    for i, (key, name) in enumerate(variables.items(), 1):
        print(f"{i}: {name}")
    
    while True:
        try:
            x_choice = int(input("\nEnter X-axis variable number: ").strip())
            if 1 <= x_choice <= len(var_list):
                x_key = var_list[x_choice - 1]
                x_label = variables[x_key]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Step 4: Choose Y-axis variables
    print(f"\nStep 4: Choose Y-axis Variables")
    print("-" * 31)
    
    y_variables = []
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    
    # First Y variable (required)
    print("Choose first Y-axis variable (Left Y-axis):")
    for i, (key, name) in enumerate(variables.items(), 1):
        if key != x_key:  # Don't show X variable as option
            print(f"{i}: {name}")
    
    while True:
        try:
            y1_choice = int(input("Enter first Y variable number: ").strip())
            if 1 <= y1_choice <= len(var_list):
                y1_key = var_list[y1_choice - 1]
                if y1_key != x_key:
                    y1_label = variables[y1_key]
                    y_variables.append({
                        'key': y1_key,
                        'label': y1_label,
                        'color': colors[0],
                        'axis': 'left'
                    })
                    break
                else:
                    print("Cannot use same variable for both X and Y axis.")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Additional Y variables (optional)
    while len(y_variables) < 5:  # Limit to 5 variables max
        add_more = input(f"\nAdd another Y variable? (y/n): ").strip().lower()
        if add_more != 'y':
            break
            
        # Choose axis side if this is the second variable
        axis_side = 'left'
        if len(y_variables) == 1:
            side_choice = input("Plot on left (l) or right (r) Y-axis? ").strip().lower()
            if side_choice == 'r':
                axis_side = 'right'
        else:
            axis_side = 'left'  # Additional variables go on left by default
        
        print(f"\nChoose next Y variable (will be plotted on {axis_side} axis):")
        available_vars = [(k, v) for k, v in variables.items() 
                         if k != x_key and k not in [yv['key'] for yv in y_variables]]
        
        for i, (key, name) in enumerate(available_vars, 1):
            print(f"{i}: {name}")
        
        if not available_vars:
            print("No more variables available.")
            break
            
        try:
            yn_choice = int(input("Enter Y variable number: ").strip())
            if 1 <= yn_choice <= len(available_vars):
                yn_key, yn_label = available_vars[yn_choice - 1]
                y_variables.append({
                    'key': yn_key,
                    'label': yn_label,
                    'color': colors[len(y_variables)],
                    'axis': axis_side
                })
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Step 5: Create the plot
    print(f"\nStep 5: Creating Plot...")
    print("-" * 22)
    
    # Extract and mask data
    x_data = selected_data[x_key][time_mask]
    
    # Create figure and axes
    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = None
    
    # Track what's plotted for legend
    all_lines = []
    all_labels = []
    
    # Plot each Y variable
    for y_var in y_variables:
        y_data = selected_data[y_var['key']][time_mask]
        
        print(f"Plotting {y_var['key']}: range {y_data.min():.3f} to {y_data.max():.3f}")
        
        if y_var['axis'] == 'left':
            line, = ax_left.plot(x_data, y_data, color=y_var['color'], 
                               label=y_var['label'])
            all_lines.append(line)
            all_labels.append(y_var['label'])
            
        else:  # right axis
            if ax_right is None:
                ax_right = ax_left.twinx()
            line, = ax_right.plot(x_data, y_data, color=y_var['color'], 
                                label=y_var['label'])
            all_lines.append(line)
            all_labels.append(y_var['label'])
    
    # Configure axes
    ax_left.set_xlabel(x_label, fontsize=12)
    
    # Set left Y-axis properties
    left_vars = [yv for yv in y_variables if yv['axis'] == 'left']
    if len(left_vars) == 1:
        ax_left.set_ylabel(left_vars[0]['label'], fontsize=12, color=left_vars[0]['color'])
        ax_left.tick_params(axis='y', labelcolor=left_vars[0]['color'])
    else:
        ax_left.set_ylabel("Left Y-axis", fontsize=12)
    
    # Set right Y-axis properties
    right_vars = [yv for yv in y_variables if yv['axis'] == 'right']
    if ax_right is not None and len(right_vars) == 1:
        ax_right.set_ylabel(right_vars[0]['label'], fontsize=12, color=right_vars[0]['color'])
        ax_right.tick_params(axis='y', labelcolor=right_vars[0]['color'])
    elif ax_right is not None:
        ax_right.set_ylabel("Right Y-axis", fontsize=12)
    
    # Create legend
    leg = fig.legend(
        all_lines, all_labels,
        loc='lower right',
        frameon=True,
        fancybox=True
    )
    
    frame = leg.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_alpha(1.0)

    
    # Create title
    y_names = " & ".join([yv['label'] for yv in y_variables])
    title = f"{y_names} vs {x_label}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Adjust layout
    fig.tight_layout()
    
# =============================================================================
#     ax_left.grid(True)
#     if ax_right is not None:
#         ax_right.grid(True)
# =============================================================================

    
    # Step 6: Save the plot
    print(f"\nStep 6: Saving Plot...")
    print("-" * 20)
    
    # Create filename
    y_keys = "_".join([yv['key'] for yv in y_variables])
    base_name = f"{y_keys}_vs_{x_key}_{selected_freq:.2e}Hz_{cycles}cycles.png"
    
    save_path = os.path.join(save_folder, base_name)
    counter = 1
    while os.path.exists(save_path):
        name_parts = base_name.split('.png')
        save_path = os.path.join(save_folder, f"{name_parts[0]}_{counter}.png")
        counter += 1
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Plot saved as: {save_path}")
    print(f"✓ Plotted {len(y_variables)} variables over {cycles} cycles")
    print("="*60)
    print(f"Plot saved as: {save_path}")
    
# ===== Run the script =====
folder = r"C:\Users\alexj\OneDrive - Drexel University\School\Research\Python\VTH\Dynamic Simulation Excel Files"
filepath = choose_excel_file(folder)

if filepath:
    print(f"\nLoading results from {filepath}...")
    dyn_results = load_dyn_results_from_excel(filepath)
    interactive_plot_xy(dyn_results)
