import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def generate_aggregated_plots(data_folder_name="experiment_data"):
    """
    Regenerates ALL graphs, including the new MASTER GROUPED BAR CHART.
    """
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    data_folder = os.path.join(project_root, data_folder_name)

    if not os.path.exists(data_folder):
        print(f"Data folder not found at: {data_folder}")
        return

    # Structure: experiments[exp_label][audio_filename][target_cm] = [response1, response2...]
    experiments = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Structure for Error Analysis: raw_data_flat[exp_label] = [ {target, response, abs_error...} ]
    raw_data_flat = defaultdict(list)

    files_processed = 0

    for filename in os.listdir(data_folder):
        if not filename.endswith(".csv"):
            continue
            
        filepath = os.path.join(data_folder, filename)
        files_processed += 1
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
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
                            
                            raw_data_flat[current_exp_label].append({
                                'target': target,
                                'response': response,
                                'signed_error': response - target,
                                'abs_error': abs(response - target)
                            })
                            
                        except (ValueError, IndexError):
                            continue
                        
        except Exception as e:
            print(f"Skipping corrupt file {filename}: {e}")

    print(f"Processed {files_processed} CSV files.")

    # --- GENERATE GRAPHS ---
    
    # 1. Standard Line Graphs
    for exp_label, audio_groups in experiments.items():
        _plot_multi_line_experiment(exp_label, audio_groups, data_folder)

    # 2. Comparisons
    _plot_master_comparison(experiments, data_folder)
    _plot_method_comparison(experiments, data_folder, "Distance", "2. Distance Single", "3. Distance Dual")
    _plot_method_comparison(experiments, data_folder, "Elevation", "4. Elevation Single", "5. Elevation Dual")

    # 3. Overall Summaries
    _plot_master_absolute_error(raw_data_flat, data_folder)
    _plot_master_signed_error(raw_data_flat, data_folder)
    
    # 4. Detailed Boxplots per Experiment
    for exp_label in raw_data_flat.keys():
        _plot_detailed_error_vs_position(exp_label, raw_data_flat[exp_label], data_folder)

    # 5. NEW: Master Grouped Bar Chart (All Dims side-by-side)
    _plot_master_grouped_position_error(raw_data_flat, data_folder)


# ==============================================================================
# NEW: MASTER GROUPED POSITION PLOT
# ==============================================================================
def _plot_master_grouped_position_error(raw_data_flat, folder):
    """
    Creates a massive grouped bar chart.
    X-Axis: Position (-300, -275, ... 300)
    Y-Axis: Median Absolute Error
    Bars: Clustered by Experiment Type (Dim) at each X.
    """
    if not raw_data_flat: return
    
    # 1. Organize data: data_map[position][experiment] = median_error
    data_map = defaultdict(dict)
    all_positions = set()
    all_experiments = sorted(raw_data_flat.keys())
    
    for exp_name in all_experiments:
        # Group errors by position for this experiment
        pos_errors = defaultdict(list)
        for item in raw_data_flat[exp_name]:
            pos_errors[item['target']].append(item['abs_error'])
        
        # Calculate median for each position
        for pos, errors in pos_errors.items():
            data_map[pos][exp_name] = np.median(errors)
            all_positions.add(pos)
            
    sorted_positions = sorted(list(all_positions))
    if not sorted_positions: return

    # 2. Setup Plot
    plt.figure(figsize=(16, 8))
    
    # Configuration for bar width
    num_exps = len(all_experiments)
    total_width = 20.0 # Space between 25cm ticks is 25, so we use 20 max
    bar_width = total_width / num_exps
    
    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, num_exps))
    
    # 3. Plot Bars
    for i, exp_name in enumerate(all_experiments):
        short_name = _shorten_name(exp_name)
        
        # X coordinates for this specific bar group
        # Shift them so they center around the tick
        # e.g. if 3 bars: offsets -1, 0, +1
        offset = (i - num_exps/2 + 0.5) * bar_width
        
        x_vals = []
        y_vals = []
        
        for pos in sorted_positions:
            if exp_name in data_map[pos]:
                x_vals.append(pos + offset)
                y_vals.append(data_map[pos][exp_name])
            else:
                # Handle missing data (e.g., Azimuth might not have -300)
                pass
                
        plt.bar(x_vals, y_vals, width=bar_width, label=short_name, color=colors[i], align='center', alpha=0.9, edgecolor='white')

    # 4. Styling
    plt.title("MASTER COMPARISON: Spatial Accuracy by Position (Side-by-Side)", fontsize=16)
    plt.xlabel("Target Position (cm)", fontsize=12)
    plt.ylabel("Median Absolute Error (cm)", fontsize=12)
    
    # Set X-Ticks to be the exact positions
    plt.xticks(sorted_positions, [int(p) for p in sorted_positions], rotation=45, fontsize=9)
    plt.xlim(min(sorted_positions)-20, max(sorted_positions)+20)
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(title="Dimension", fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(folder, "GRAPH_MASTER_DETAILED_COMPARISON.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" -> Generated Master Grouped Graph: {save_path}")


# ==============================================================================
# HELPERS
# ==============================================================================

def _plot_detailed_error_vs_position(exp_name, data_list, folder):
    grouped_signed = defaultdict(list)
    grouped_abs = defaultdict(list)
    
    for item in data_list:
        t = item['target']
        grouped_signed[t].append(item['signed_error'])
        grouped_abs[t].append(item['abs_error'])
        
    sorted_targets = sorted(grouped_signed.keys())
    if not sorted_targets: return

    short_name = _shorten_name(exp_name)
    safe_name = "".join([c for c in short_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")

    # SIGNED
    plt.figure(figsize=(14, 7))
    plt.axhline(y=0, color='black', alpha=0.8) 
    plt.boxplot([grouped_signed[t] for t in sorted_targets], positions=sorted_targets, widths=15, 
                patch_artist=True, showfliers=False, 
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    plt.title(f"BIAS: {short_name}", fontsize=14)
    plt.xlabel("Target (cm)")
    plt.ylabel("Signed Error (cm)")
    plt.xticks(sorted_targets, [int(t) for t in sorted_targets], rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(folder, f"GRAPH_DETAIL_SIGNED_{safe_name}.png"))
    plt.close()

    # ABSOLUTE
    plt.figure(figsize=(14, 7))
    plt.boxplot([grouped_abs[t] for t in sorted_targets], positions=sorted_targets, widths=15, 
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='salmon'), medianprops=dict(color='darkred'))
    plt.title(f"PRECISION: {short_name}", fontsize=14)
    plt.xlabel("Target (cm)")
    plt.ylabel("Abs Error (cm)")
    plt.xticks(sorted_targets, [int(t) for t in sorted_targets], rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(folder, f"GRAPH_DETAIL_ABS_{safe_name}.png"))
    plt.close()

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
        means = [np.mean(data_dict[t]) for t in targets]
        stds = [np.std(data_dict[t]) for t in targets]
        clean_name = os.path.basename(audio_name)
        plt.errorbar(targets, means, yerr=stds, fmt='o-', capsize=4, label=clean_name, color=colors[i], alpha=0.8)
    plt.title(f"{exp_name}")
    plt.legend()
    plt.grid(True, linestyle=':')
    safe_name = "".join([c for c in exp_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    plt.savefig(os.path.join(folder, f"GRAPH_{safe_name}.png"))
    plt.close()

def _plot_master_comparison(experiments, folder):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Dark2(np.linspace(0, 1, len(experiments)))
    has_data = False
    for i, (exp_name, audio_groups) in enumerate(experiments.items()):
        merged = defaultdict(list)
        for _, data_dict in audio_groups.items():
            for t, r in data_dict.items():
                merged[t].extend(r)
        if not merged: continue
        has_data = True
        targets = sorted(merged.keys())
        means = [np.mean(merged[t]) for t in targets]
        plt.plot(targets, means, 'o-', linewidth=2, label=_shorten_name(exp_name), color=colors[i])
    if has_data:
        plt.title("MASTER: Position Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder, "GRAPH_MASTER_POSITIONS.png"))
    plt.close()

def _plot_method_comparison(experiments, folder, dim_name, key_single, key_dual):
    label_single = next((k for k in experiments.keys() if key_single in k), None)
    label_dual = next((k for k in experiments.keys() if key_dual in k), None)
    if not label_single or not label_dual: return
    plt.figure(figsize=(10, 6))
    for label, color, tag in [(label_single, 'blue', 'Single'), (label_dual, 'red', 'Dual')]:
        merged = defaultdict(list)
        for _, data_dict in experiments[label].items():
            for t, r in data_dict.items(): merged[t].extend(r)
        if not merged: continue
        targets = sorted(merged.keys())
        means = [np.mean(merged[t]) for t in targets]
        stds = [np.std(merged[t]) for t in targets]
        plt.errorbar(targets, means, yerr=stds, fmt='o-', capsize=5, label=tag, color=color)
    plt.title(f"COMPARE: {dim_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder, f"GRAPH_COMPARE_{dim_name.upper()}.png"))
    plt.close()

def _plot_master_absolute_error(raw_data, folder):
    if not raw_data: return
    plt.figure(figsize=(12, 7))
    exp_names = sorted(raw_data.keys())
    means = []
    sems = []
    labels = []
    for exp in exp_names:
        abs_errors = [d['abs_error'] for d in raw_data[exp]]
        if not abs_errors: continue
        means.append(np.mean(abs_errors))
        sems.append(np.std(abs_errors) / np.sqrt(len(abs_errors)))
        labels.append(_shorten_name(exp))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=10, color='skyblue', alpha=0.7)
    plt.xticks(x, labels, rotation=15)
    plt.title('MASTER: Mean Absolute Error')
    plt.savefig(os.path.join(folder, "GRAPH_MASTER_ERROR_ABSOLUTE.png"))
    plt.close()

def _plot_master_signed_error(raw_data, folder):
    if not raw_data: return
    plt.figure(figsize=(12, 7))
    exp_names = sorted(raw_data.keys())
    data = [[d['signed_error'] for d in raw_data[exp]] for exp in exp_names if raw_data[exp]]
    labels = [_shorten_name(exp) for exp in exp_names if raw_data[exp]]
    plt.axhline(0, color='black')
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.xticks(rotation=15)
    plt.title('MASTER: Signed Error (Bias)')
    plt.savefig(os.path.join(folder, "GRAPH_MASTER_ERROR_SIGNED.png"))
    plt.close()

def _shorten_name(exp_name):
    if '.' in exp_name:
        parts = exp_name.split('.')
        if len(parts) > 1:
            return parts[1].split('(')[0].strip()
    return exp_name