import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def generate_aggregated_plots(data_folder_name="experiment_data"):
    """
    1. Scans all CSVs (old and new).
    2. Generates individual graphs for each Experiment (split by Audio).
    3. Generates a MASTER GRAPH comparing all Dimensions against each other.
    """
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    data_folder = os.path.join(project_root, data_folder_name)

    if not os.path.exists(data_folder):
        print(f"Data folder not found at: {data_folder}")
        return

    # Structure: experiments[exp_label][audio_filename][target_cm] = [response1, response2...]
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
                
                current_exp = None
                current_audio = "Unknown Audio"
                parsing_data = False
                
                for row in reader:
                    if not row: continue 
                    
                    # 1. Detect Metadata
                    if len(row) >= 2:
                        key = row[0].strip()
                        val = row[1].strip()
                        if key == "Experiment":
                            current_exp = val
                            parsing_data = False 
                        elif key == "Audio":
                            current_audio = val
                    
                    # 2. Detect Data Block
                    if len(row) > 0 and row[0] == "Trial_ID":
                        parsing_data = True
                        continue 
                        
                    # 3. Parse Data
                    if parsing_data and current_exp:
                        try:
                            if len(row) < 4: continue
                            target = float(row[2])
                            response = float(row[3])
                            experiments[current_exp][current_audio][target].append(response)
                        except (ValueError, IndexError):
                            continue
                        
        except Exception as e:
            print(f"Skipping corrupt file {filename}: {e}")

    print(f"Processed {files_processed} CSV files.")

    # --- STEP 2: Generate Individual Experiment Plots ---
    for exp_name, audio_groups in experiments.items():
        _plot_multi_line_experiment(exp_name, audio_groups, data_folder)

    # --- STEP 3: Generate Master Comparison Plot ---
    _plot_master_comparison(experiments, data_folder)


def _plot_multi_line_experiment(exp_name, audio_groups, folder):
    """Plots one Experiment, with separate lines for each Audio File."""
    plt.figure(figsize=(10, 6))
    
    all_targets = []
    for targets_dict in audio_groups.values():
        all_targets.extend(targets_dict.keys())
        
    if not all_targets: 
        plt.close()
        return

    min_val = min(all_targets) - 25
    max_val = max(all_targets) + 25
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label="Ideal Target")
    
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

    plt.title(f"Results: {exp_name}")
    plt.xlabel("Target Position (cm)")
    plt.ylabel("Perceived Position (cm)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Stimuli")
    plt.axis('equal')
    
    safe_name = "".join([c for c in exp_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    save_path = os.path.join(folder, f"GRAPH_{safe_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f" -> Generated Graph: {save_path}")


def _plot_master_comparison(experiments, folder):
    """
    Aggregates ALL data for each Experiment Type (ignoring audio differences)
    and plots them on a single graph for comparison.
    """
    plt.figure(figsize=(12, 8))
    
    # Track bounds for the diagonal line
    global_min = 0
    global_max = 0
    has_data = False

    # Use a distinct color for each Experiment Type
    colors = plt.cm.Dark2(np.linspace(0, 1, len(experiments)))

    for i, (exp_name, audio_groups) in enumerate(experiments.items()):
        # 1. Merge all audio data for this experiment into one bucket
        merged_data = defaultdict(list)
        
        for audio_file, data_dict in audio_groups.items():
            for target, responses in data_dict.items():
                merged_data[target].extend(responses)
        
        if not merged_data: continue
        has_data = True
        
        # 2. Calculate Stats
        targets = sorted(merged_data.keys())
        means = []
        # specific standard deviation for error bars
        # stds = [] 
        
        for t in targets:
            responses = merged_data[t]
            means.append(np.mean(responses))
            # stds.append(np.std(responses))
        
        # Update bounds
        global_min = min(global_min, min(targets))
        global_max = max(global_max, max(targets))

        # 3. Plot Line
        # Shorten name for legend (e.g., "1. Azimuth (Left/Right)" -> "Azimuth")
        short_name = exp_name.split('.')[1].split('(')[0].strip() if '.' in exp_name else exp_name
        
        plt.plot(targets, means, 'o-', linewidth=2, label=short_name, color=colors[i], markersize=6)
        
        # Optional: Add faint error band instead of messy bars
        # plt.fill_between(targets, np.array(means)-np.array(stds), np.array(means)+np.array(stds), color=colors[i], alpha=0.1)

    if not has_data:
        plt.close()
        return

    # Add Ideal Line
    pad = 25
    plt.plot([global_min - pad, global_max + pad], [global_min - pad, global_max + pad], 
             'k--', alpha=0.3, linewidth=1, label="Ideal")

    plt.title("MASTER COMPARISON: Localization Accuracy by Dimension", fontsize=14)
    plt.xlabel("Target Position (cm)", fontsize=12)
    plt.ylabel("Perceived Position (cm)", fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.legend(title="Dimension", fontsize=10)
    plt.axis('equal')
    
    save_path = os.path.join(folder, "GRAPH_MASTER_COMPARISON.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" -> Generated Master Graph: {save_path}")