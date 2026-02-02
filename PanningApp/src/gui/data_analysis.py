import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def generate_aggregated_plots(data_folder_name="experiment_data"):
    """
    1. Scans all CSVs.
    2. Generates detailed graphs for each Experiment.
    3. Generates a MASTER GRAPH (All dims).
    4. Generates COMPARISON GRAPHS (Mono vs Dual for Dist/Elev).
    """
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    data_folder = os.path.join(project_root, data_folder_name)

    if not os.path.exists(data_folder):
        print(f"Data folder not found at: {data_folder}")
        return

    # Structure: experiments[exp_key][audio_filename][target_cm] = [response1, response2...]
    experiments = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    files_processed = 0

    for filename in os.listdir(data_folder):
        if not filename.endswith(".csv"):
            continue
            
        filepath = os.path.join(data_folder, filename)
        files_processed += 1
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # We need to map the readable Label back to the Key if possible, 
                # OR just use the Label as the key. 
                # Since experiment_logic saves the LABEL in the CSV, we use the Label.
                current_exp_label = None
                current_audio = "Unknown Audio"
                parsing_data = False
                
                for row in reader:
                    if not row: continue 
                    
                    if len(row) >= 2:
                        key = row[0].strip()
                        val = row[1].strip()
                        if key == "Experiment":
                            current_exp_label = val
                            parsing_data = False 
                        elif key == "Audio":
                            current_audio = val
                    
                    if len(row) > 0 and row[0] == "Trial_ID":
                        parsing_data = True
                        continue 
                        
                    if parsing_data and current_exp_label:
                        try:
                            if len(row) < 4: continue
                            target = float(row[2])
                            response = float(row[3])
                            experiments[current_exp_label][current_audio][target].append(response)
                        except (ValueError, IndexError):
                            continue
                        
        except Exception as e:
            print(f"Skipping corrupt file {filename}: {e}")

    print(f"Processed {files_processed} CSV files.")

    # --- 1. Individual Plots ---
    for exp_label, audio_groups in experiments.items():
        _plot_multi_line_experiment(exp_label, audio_groups, data_folder)

    # --- 2. Master Comparison ---
    _plot_master_comparison(experiments, data_folder)

    # --- 3. Mono vs Dual Comparisons ---
    # We look for substrings because the labels in CSV might vary slightly
    _plot_method_comparison(experiments, data_folder, 
                            "Distance", 
                            "2. Distance Single", 
                            "3. Distance Dual")
                            
    _plot_method_comparison(experiments, data_folder, 
                            "Elevation", 
                            "4. Elevation Single", 
                            "5. Elevation Dual")


def _plot_multi_line_experiment(exp_name, audio_groups, folder):
    plt.figure(figsize=(10, 6))
    
    all_targets = []
    for targets_dict in audio_groups.values():
        all_targets.extend(targets_dict.keys())
        
    if not all_targets: 
        plt.close()
        return

    min_val = min(all_targets) - 25
    max_val = max(all_targets) + 25
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label="Ideal")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(audio_groups)))
    
    for i, (audio_name, data_dict) in enumerate(audio_groups.items()):
        targets = sorted(data_dict.keys())
        means = []
        stds = []
        for t in targets:
            responses = data_dict[t]
            means.append(np.mean(responses))
            stds.append(np.std(responses))
            
        clean_name = os.path.basename(audio_name)
        plt.errorbar(targets, means, yerr=stds, fmt='o-', capsize=4, label=clean_name, color=colors[i], alpha=0.8)

    plt.title(f"{exp_name}")
    plt.xlabel("Target (cm)")
    plt.ylabel("Response (cm)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Stimuli")
    plt.axis('equal')
    
    safe_name = "".join([c for c in exp_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    plt.savefig(os.path.join(folder, f"GRAPH_{safe_name}.png"))
    plt.close()


def _plot_master_comparison(experiments, folder):
    plt.figure(figsize=(12, 8))
    global_min = 0
    global_max = 0
    has_data = False
    colors = plt.cm.Dark2(np.linspace(0, 1, len(experiments)))

    for i, (exp_name, audio_groups) in enumerate(experiments.items()):
        # Merge all audio
        merged = defaultdict(list)
        for _, data_dict in audio_groups.items():
            for t, r in data_dict.items():
                merged[t].extend(r)
        
        if not merged: continue
        has_data = True
        
        targets = sorted(merged.keys())
        means = [np.mean(merged[t]) for t in targets]
        
        global_min = min(global_min, min(targets))
        global_max = max(global_max, max(targets))

        short_name = exp_name.split('.')[1].split('(')[0].strip() if '.' in exp_name else exp_name
        plt.plot(targets, means, 'o-', linewidth=2, label=short_name, color=colors[i])

    if has_data:
        pad = 25
        plt.plot([global_min - pad, global_max + pad], [global_min - pad, global_max + pad], 'k--', alpha=0.3)
        plt.title("MASTER COMPARISON: All Dimensions")
        plt.xlabel("Target (cm)")
        plt.ylabel("Response (cm)")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.savefig(os.path.join(folder, "GRAPH_MASTER_COMPARISON.png"))
    plt.close()


def _plot_method_comparison(experiments, folder, dim_name, key_single, key_dual):
    """
    Finds experiments matching 'key_single' and 'key_dual' and plots them 
    on one graph to compare Mono vs Dual Mono.
    """
    # 1. Find the actual full keys (labels) used in the dictionary
    label_single = next((k for k in experiments.keys() if key_single in k), None)
    label_dual = next((k for k in experiments.keys() if key_dual in k), None)

    if not label_single or not label_dual:
        print(f"Skipping {dim_name} comparison (Missing data for {key_single} or {key_dual})")
        return

    plt.figure(figsize=(10, 6))
    
    # 2. Extract and Plot Data
    for label, color, tag in [(label_single, 'blue', 'Single (Mono)'), (label_dual, 'red', 'Dual (Phantom)')]:
        
        # Merge all audio files for this method
        merged = defaultdict(list)
        for _, data_dict in experiments[label].items():
            for t, r in data_dict.items():
                merged[t].extend(r)
                
        if not merged: continue
        
        targets = sorted(merged.keys())
        means = [np.mean(merged[t]) for t in targets]
        stds = [np.std(merged[t]) for t in targets]
        
        # Plot
        plt.errorbar(targets, means, yerr=stds, fmt='o-', capsize=5, 
                     label=tag, color=color, linewidth=2, alpha=0.8)
        
        # Determine bounds
        min_val = min(targets) - 25
        max_val = max(targets) + 25
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.1) # Faint reference

    plt.title(f"METHOD COMPARISON: {dim_name} (Mono vs. Dual)")
    plt.xlabel("Target Position (cm)")
    plt.ylabel("Perceived Position (cm)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    
    save_path = os.path.join(folder, f"GRAPH_COMPARE_{dim_name.upper()}.png")
    plt.savefig(save_path)
    plt.close()
    print(f" -> Generated Comparison: {save_path}")