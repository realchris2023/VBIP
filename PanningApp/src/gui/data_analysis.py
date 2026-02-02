import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def generate_aggregated_plots(data_folder="experiment_data"):
    """
    Scans the data folder, aggregates data by Experiment Type,
    and generates a summary plot for each experiment type.
    """
    if not os.path.exists(data_folder):
        return

    # 1. Collect Data
    # Structure: experiments[exp_label][target_cm] = [response1, response2, ...]
    experiments = defaultdict(lambda: defaultdict(list))
    
    for filename in os.listdir(data_folder):
        if not filename.endswith(".csv"):
            continue
            
        filepath = os.path.join(data_folder, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # Parse Header to find Experiment Type
                # Row 0: ["Experiment", "Label"]
                if len(rows) < 10: continue # Skip empty/corrupt files
                
                exp_label = rows[0][1] # e.g. "1. Azimuth (Left / Right)"
                
                # Find the start of data (Look for header "Trial_ID")
                start_row_idx = 0
                for i, row in enumerate(rows):
                    if row and row[0] == "Trial_ID":
                        start_row_idx = i + 1
                        break
                
                # Parse Data Rows
                for row in rows[start_row_idx:]:
                    if not row: continue
                    try:
                        # Row Format: ID, Side, Target, Response, Error, Notes
                        target = float(row[2])
                        response = float(row[3])
                        
                        experiments[exp_label][target].append(response)
                    except ValueError:
                        continue
                        
        except Exception as e:
            print(f"Skipping corrupt file {filename}: {e}")

    # 2. Generate Plots
    for exp_name, data_points in experiments.items():
        _plot_single_experiment(exp_name, data_points, data_folder)

def _plot_single_experiment(exp_name, data_dict, folder):
    """
    Generates a Mean vs Expected plot with Error Bars.
    """
    targets = sorted(data_dict.keys())
    means = []
    stds = []
    
    for t in targets:
        responses = data_dict[t]
        means.append(np.mean(responses))
        stds.append(np.std(responses))
    
    # Setup Plot
    plt.figure(figsize=(10, 6))
    
    # 1. Plot Ideal Line (y=x)
    min_val = min(targets) - 25
    max_val = max(targets) + 25
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Ideal (Target)")
    
    # 2. Plot Data (Mean with Error Bars)
    # fmt='o' makes it a scatter plot with bars
    plt.errorbar(targets, means, yerr=stds, fmt='o-', capsize=5, color='blue', label="Mean Response ± StdDev")
    
    # Formatting
    plt.title(f"Aggregated Results: {exp_name}")
    plt.xlabel("Target Position (cm)")
    plt.ylabel("Perceived Position (cm)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Make axes symmetric
    plt.axis('equal')
    
    # Save
    # Sanitize filename
    safe_name = "".join([c for c in exp_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    save_path = os.path.join(folder, f"GRAPH_{safe_name}.png")
    
    plt.savefig(save_path)
    plt.close() # Close memory
    print(f"Generated Graph: {save_path}")