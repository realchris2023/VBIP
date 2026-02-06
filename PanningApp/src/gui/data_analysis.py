import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# --- 1. GLOBAL VISUAL SETTINGS ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'figure.autolayout': True
})

try:
    from scipy import stats 
except ImportError:
    stats = None
    print("WARNING: scipy not found. T-Test will be skipped.")

# --- CONFIGURATION: EXPERIMENT RADII ---
RADII = {
    "Azimuth": 212.0,
    "Elevation": 212.0,
    "Distance": 145.0  
}

def generate_aggregated_plots(data_folder_name="experiment_data"):
    
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    data_folder = os.path.join(project_root, data_folder_name)

    if not os.path.exists(data_folder):
        print(f"Data folder not found at: {data_folder}")
        return

    # Data Containers
    experiments = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    raw_data_flat = defaultdict(list)
    participant_scores = defaultdict(lambda: defaultdict(list))

    files_processed = 0

    for filename in os.listdir(data_folder):
        if not filename.endswith(".csv"): continue
        filepath = os.path.join(data_folder, filename)
        files_processed += 1
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                current_exp_label = None
                current_audio = "Unknown"
                current_pid = "Unknown"
                parsing_data = False
                
                for row in reader:
                    if not row: continue 
                    if len(row) >= 2:
                        key, val = row[0].strip(), row[1].strip()
                        if key == "Experiment":
                            current_exp_label = val
                            parsing_data = False 
                        elif key == "Audio": current_audio = val
                        elif key == "Participant": current_pid = val
                    
                    if len(row) > 0 and row[0] == "Trial_ID":
                        parsing_data = True
                        continue 
                    
                    if parsing_data and current_exp_label:
                        try:
                            if len(row) < 4: continue
                            target = float(row[2])
                            response = float(row[3])
                            abs_err = abs(response - target)
                            
                            experiments[current_exp_label][current_audio][target].append(response)
                            raw_data_flat[current_exp_label].append({
                                'target': target, 'response': response,
                                'signed_error': response - target, 'abs_error': abs_err
                            })
                            
                            if current_pid != "Unknown":
                                participant_scores[current_exp_label][current_pid].append(abs_err)
                            
                        except (ValueError, IndexError): continue
        except Exception: pass

    print(f"Processed {files_processed} CSV files.")

    # --- 1. GENERATE GRAPHS (CM) ---
    for exp_label, audio_groups in experiments.items():
        _plot_multi_line_experiment(exp_label, audio_groups, data_folder, use_deg=False)

    _plot_master_comparison(experiments, data_folder, use_deg=False)
    _plot_method_comparison(experiments, data_folder, "Distance", "2. Distance Single", "3. Distance Dual", use_deg=False)
    _plot_method_comparison(experiments, data_folder, "Elevation", "4. Elevation Single", "5. Elevation Dual", use_deg=False)
    _plot_master_absolute_error(raw_data_flat, data_folder, use_deg=False)
    _plot_master_grouped_position_error(raw_data_flat, data_folder, use_deg=False)
    _plot_master_signed_error(raw_data_flat, data_folder, use_deg=False)
    
    for exp_label in raw_data_flat.keys():
        _plot_detailed_error_vs_position(exp_label, raw_data_flat[exp_label], data_folder, use_deg=False)

    # --- 2. GENERATE GRAPHS (DEGREES) ---
    print("Generating Angular (Degree) Graphs...")
    
    for exp_label, audio_groups in experiments.items():
        _plot_multi_line_experiment(exp_label, audio_groups, data_folder, use_deg=True)

    _plot_master_comparison(experiments, data_folder, use_deg=True)
    _plot_method_comparison(experiments, data_folder, "Distance", "2. Distance Single", "3. Distance Dual", use_deg=True)
    _plot_method_comparison(experiments, data_folder, "Elevation", "4. Elevation Single", "5. Elevation Dual", use_deg=True)
    
    # NOTE: We keep Mean for this ONE summary graph to match the T-Test
    _plot_master_absolute_error(raw_data_flat, data_folder, use_deg=True)
    
    _plot_master_grouped_position_error(raw_data_flat, data_folder, use_deg=True)
    _plot_master_signed_error(raw_data_flat, data_folder, use_deg=True)

    for exp_label in raw_data_flat.keys():
        _plot_detailed_error_vs_position(exp_label, raw_data_flat[exp_label], data_folder, use_deg=True)

    # --- 3. REPORTS ---
    _generate_leaderboard_file(participant_scores, data_folder)
    if stats:
        _generate_statistical_report(raw_data_flat, data_folder)


# ==============================================================================
# HELPERS
# ==============================================================================
def _get_radius_for_exp(exp_name):
    if "Azimuth" in exp_name: return RADII["Azimuth"]
    if "Elevation" in exp_name: return RADII["Elevation"]
    if "Distance" in exp_name: return RADII["Distance"]
    return 200.0 

def _cm_to_deg(cm_val, radius):
    return np.degrees(np.arctan(cm_val / radius))

def _shorten_name(exp_name):
    if '.' in exp_name:
        parts = exp_name.split('.')
        if len(parts) > 1: return parts[1].split('(')[0].strip()
    return exp_name

def _add_reference_lines(exp_name, use_deg):
    """
    Adds vertical lines for Speakers (Gray) and Room Boundaries (Red).
    """
    radius = _get_radius_for_exp(exp_name)
    
    speakers = []
    boundaries = []
    boundary_labels = []

    # --- DEFINE GEOMETRY ---
    
    # 1. ELEVATION (Single & Dual)
    if "Elevation" in exp_name:
        speakers = [-92.5, 92.5]
        boundaries = [-127.5, 137.5] # Floor and Ceiling
        boundary_labels = ["Floor", "Ceiling"]
        
    # 2. AZIMUTH
    elif "Azimuth" in exp_name:
        speakers = [-122.5, 122.5]
        boundaries = [] 
        
    # 3. DISTANCE DUAL (Dual Mono)
    elif "Distance" in exp_name and "Dual" in exp_name:
        speakers = [-122.5, 122.5]
        boundaries = [-122.5] 
        boundary_labels = ["Rear Wall"]
        
    # 4. DISTANCE SINGLE
    elif "Distance" in exp_name and "Single" in exp_name:
        speakers = [-122.5, 122.5] # Speakers are still there
        boundaries = [] 
        
    else:
        # Fallback if we can't determine type
        return

    # --- PLOT SPEAKERS (Gray Dashed) ---
    for pos in speakers:
        val = _cm_to_deg(pos, radius) if use_deg else pos
        plt.axvline(x=val, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        # Label at top
        plt.text(val, plt.ylim()[1], 'Spk', rotation=90, verticalalignment='top', color='gray', fontsize=10)

    # --- PLOT BOUNDARIES (Red Dotted) ---
    for i, pos in enumerate(boundaries):
        val = _cm_to_deg(pos, radius) if use_deg else pos
        label = boundary_labels[i] if i < len(boundary_labels) else "Wall"
        
        plt.axvline(x=val, color='red', linestyle=':', alpha=0.6, linewidth=2.0)
        # Label at bottom
        plt.text(val, plt.ylim()[0], label, rotation=90, verticalalignment='bottom', color='red', fontsize=10)


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def _plot_multi_line_experiment(exp_name, audio_groups, folder, use_deg=False):
    plt.figure(figsize=(10, 6))
    all_targets = []
    for targets_dict in audio_groups.values(): all_targets.extend(targets_dict.keys())
    if not all_targets: 
        plt.close()
        return
        
    radius = _get_radius_for_exp(exp_name)
    suffix = "_DEG" if use_deg else ""
    unit = "Degrees" if use_deg else "cm"
    
    min_t, max_t = min(all_targets), max(all_targets)
    if use_deg:
        min_d = _cm_to_deg(min_t, radius)
        max_d = _cm_to_deg(max_t, radius)
        plt.plot([min_d, max_d], [min_d, max_d], 'k--', alpha=0.3, label="Ideal", linewidth=2)
    else:
        plt.plot([min_t, max_t], [min_t, max_t], 'k--', alpha=0.3, label="Ideal", linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(audio_groups)))
    
    for i, (audio, data) in enumerate(audio_groups.items()):
        targets = sorted(data.keys())
        medians = [] 
        stds = []
        x_vals = []
        
        for t in targets:
            responses = data[t]
            if use_deg:
                x_val = _cm_to_deg(t, radius)
                resp_val = [_cm_to_deg(r, radius) for r in responses]
            else:
                x_val = t
                resp_val = responses
                
            x_vals.append(x_val)
            medians.append(np.median(resp_val))
            stds.append(np.std(resp_val))
            
        plt.errorbar(x_vals, medians, yerr=stds, fmt='o-', label=os.path.basename(audio), capsize=4, color=colors[i])
        
    _add_reference_lines(exp_name, use_deg)

    plt.title(f"{exp_name} ({unit})")
    plt.xlabel(f"Target Position ({unit})")
    plt.ylabel(f"Response (Median) [{unit}]")
    plt.legend()
    plt.grid(True, linestyle=':')
    safe = "".join([c for c in exp_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    plt.savefig(os.path.join(folder, f"GRAPH_{safe}{suffix}.png"))
    plt.close()

def _plot_master_comparison(experiments, folder, use_deg=False):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Dark2(np.linspace(0, 1, len(experiments)))
    has_data = False
    suffix = "_DEG" if use_deg else ""
    unit = "Degrees" if use_deg else "cm"

    global_min, global_max = float('inf'), float('-inf')

    for i, (exp_name, audio_groups) in enumerate(experiments.items()):
        merged = defaultdict(list)
        radius = _get_radius_for_exp(exp_name)
        
        for _, data_dict in audio_groups.items():
            for t, r in data_dict.items(): merged[t].extend(r)
        if not merged: continue
        has_data = True
        
        targets = sorted(merged.keys())
        x_vals = []
        medians = [] 
        
        for t in targets:
            responses = merged[t]
            if use_deg:
                x_val = _cm_to_deg(t, radius)
                med_val = np.median([_cm_to_deg(r, radius) for r in responses])
            else:
                x_val = t
                med_val = np.median(responses)
            
            x_vals.append(x_val)
            medians.append(med_val)
            global_min = min(global_min, x_val)
            global_max = max(global_max, x_val)
            
        plt.plot(x_vals, medians, 'o-', linewidth=2, label=_shorten_name(exp_name), color=colors[i])

    if has_data:
        pad = 5 if use_deg else 25
        plt.plot([global_min-pad, global_max+pad], [global_min-pad, global_max+pad], 'k--', alpha=0.3, label="Ideal")
        
        plt.title(f"MASTER: Position Comparison ({unit})")
        plt.xlabel(f"Target Position ({unit})")
        plt.ylabel(f"Response (Median) [{unit}]")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(folder, f"GRAPH_MASTER_POSITIONS{suffix}.png"))
    plt.close()

def _plot_method_comparison(experiments, folder, dim, k1, k2, use_deg=False):
    label_single = next((k for k in experiments.keys() if k1 in k), None)
    label_dual = next((k for k in experiments.keys() if k2 in k), None)
    if not label_single or not label_dual: return
    
    suffix = "_DEG" if use_deg else ""
    unit = "Degrees" if use_deg else "cm"
    
    plt.figure(figsize=(10, 6))
    
    all_x = []
    
    for label, color, tag in [(label_single, 'blue', 'Single'), (label_dual, 'red', 'Dual')]:
        radius = _get_radius_for_exp(label)
        merged = defaultdict(list)
        for _, data_dict in experiments[label].items():
            for t, r in data_dict.items(): merged[t].extend(r)
        if not merged: continue
        
        targets = sorted(merged.keys())
        x_vals, medians, stds = [], [], []
        
        for t in targets:
            responses = merged[t]
            if use_deg:
                x_val = _cm_to_deg(t, radius)
                vals = [_cm_to_deg(r, radius) for r in responses]
            else:
                x_val = t
                vals = responses
                
            x_vals.append(x_val)
            medians.append(np.median(vals)) 
            stds.append(np.std(vals))
            all_x.append(x_val)
            
        plt.errorbar(x_vals, medians, yerr=stds, fmt='o-', capsize=5, label=tag, color=color)

    if all_x:
        min_x, max_x = min(all_x), max(all_x)
        plt.plot([min_x, max_x], [min_x, max_x], 'k--', alpha=0.3, label="Ideal")

    # Use reference lines for single (default) or dual if active
    ref_exp = label_dual if "Dual" in label_dual else label_single
    _add_reference_lines(ref_exp, use_deg)

    plt.title(f"COMPARE: {dim} ({unit})")
    plt.xlabel(f"Target Position ({unit})")
    plt.ylabel(f"Response (Median) [{unit}]")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(folder, f"GRAPH_COMPARE_{dim.upper()}{suffix}.png"))
    plt.close()

def _plot_master_absolute_error(raw_data, folder, use_deg=False):
    if not raw_data: return
    plt.figure(figsize=(12, 7))
    exp_names = sorted(raw_data.keys())
    means, sems, labels = [], [], []
    suffix = "_DEG" if use_deg else ""
    unit = "Degrees" if use_deg else "cm"
    
    for exp in exp_names:
        radius = _get_radius_for_exp(exp)
        errors = [d['abs_error'] for d in raw_data[exp]]
        if use_deg:
            errors = [_cm_to_deg(e, radius) for e in errors]
            
        if not errors: continue
        means.append(np.mean(errors)) 
        sems.append(np.std(errors) / np.sqrt(len(errors)))
        labels.append(_shorten_name(exp))
        
    plt.bar(np.arange(len(labels)), means, yerr=sems, capsize=10, color='skyblue', alpha=0.7)
    plt.xticks(np.arange(len(labels)), labels, rotation=15)
    plt.title(f'MASTER: Mean Absolute Error ({unit})')
    plt.ylabel(f"Error ({unit})")
    plt.savefig(os.path.join(folder, f"GRAPH_MASTER_ERROR_ABSOLUTE{suffix}.png"))
    plt.close()

def _plot_master_grouped_position_error(raw_data_flat, folder, use_deg=False):
    if not raw_data_flat: return
    data_map = defaultdict(dict)
    all_positions = set()
    all_experiments = sorted(raw_data_flat.keys())
    suffix = "_DEG" if use_deg else ""
    unit = "Degrees" if use_deg else "cm"
    
    for exp_name in all_experiments:
        radius = _get_radius_for_exp(exp_name)
        pos_errors = defaultdict(list)
        for item in raw_data_flat[exp_name]:
            val = item['abs_error']
            if use_deg: val = _cm_to_deg(val, radius)
            pos_errors[item['target']].append(val)
            
        for pos, errors in pos_errors.items():
            data_map[pos][exp_name] = np.median(errors) # MEDIAN
            all_positions.add(pos)
            
    sorted_positions = sorted(list(all_positions))
    if not sorted_positions: return
    
    plt.figure(figsize=(16, 8))
    num_exps = len(all_experiments)
    bar_width = 20.0 / num_exps
    colors = plt.cm.tab10(np.linspace(0, 1, num_exps))
    
    if use_deg:
        xtick_labels = [int(p) for p in sorted_positions]
        xlabel = "Target Position (cm) - [Converted Y to Degrees]"
    else:
        xtick_labels = [int(p) for p in sorted_positions]
        xlabel = "Target Position (cm)"

    for i, exp_name in enumerate(all_experiments):
        offset = (i - num_exps/2 + 0.5) * bar_width
        x_vals = [p + offset for p in sorted_positions if exp_name in data_map[p]]
        y_vals = [data_map[p][exp_name] for p in sorted_positions if exp_name in data_map[p]]
        plt.bar(x_vals, y_vals, width=bar_width, label=_shorten_name(exp_name), color=colors[i], align='center', alpha=0.9, edgecolor='white')
        
    plt.title(f"MASTER: Accuracy by Position ({unit})")
    plt.xlabel(xlabel)
    plt.ylabel(f"Median Abs Error ({unit})")
    plt.xticks(sorted_positions, xtick_labels, rotation=45)
    plt.xlim(min(sorted_positions)-25, max(sorted_positions)+25)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"GRAPH_MASTER_DETAILED_COMPARISON{suffix}.png"))
    plt.close()

def _plot_master_signed_error(raw_data, folder, use_deg=False):
    if not raw_data: return
    plt.figure(figsize=(12, 7))
    exp_names = sorted(raw_data.keys())
    data = []
    labels = []
    suffix = "_DEG" if use_deg else ""
    unit = "Degrees" if use_deg else "cm"
    
    for exp in exp_names:
        radius = _get_radius_for_exp(exp)
        errors = [d['signed_error'] for d in raw_data[exp]]
        if use_deg:
            errors = [_cm_to_deg(e, radius) for e in errors]
            
        if not errors: continue
        data.append(errors)
        labels.append(_shorten_name(exp))
        
    plt.axhline(0, color='black')
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    plt.xticks(rotation=15)
    plt.ylabel(f"Signed Error ({unit})")
    plt.title(f'MASTER: Bias Comparison ({unit})')
    plt.savefig(os.path.join(folder, f"GRAPH_MASTER_ERROR_SIGNED{suffix}.png"))
    plt.close()

def _plot_detailed_error_vs_position(exp_name, data_list, folder, use_deg=False):
    grouped = defaultdict(list)
    radius = _get_radius_for_exp(exp_name)
    suffix = "_DEG" if use_deg else ""
    unit = "Degrees" if use_deg else "cm"
    
    for item in data_list:
        val = item['signed_error']
        if use_deg: val = _cm_to_deg(val, radius)
        grouped[item['target']].append(val)
        
    targets = sorted(grouped.keys())
    if not targets: return
    short = _shorten_name(exp_name)
    safe = "".join([c for c in short if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    
    if use_deg:
        xticklabels = [f"{_cm_to_deg(t, radius):.1f}°" for t in targets]
        xlabel = "Target (Degrees)"
    else:
        xticklabels = [int(t) for t in targets]
        xlabel = "Target (cm)"

    plt.figure(figsize=(14, 7))
    plt.axhline(0, color='black')
    plt.boxplot([grouped[t] for t in targets], positions=targets, widths=15, showfliers=False)
    
    _add_reference_lines(exp_name, use_deg)
    
    plt.title(f"BIAS: {short} [{unit}]")
    plt.xticks(targets, xticklabels, rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(f"Error ({unit})")
    plt.grid(True, linestyle=':')
    plt.savefig(os.path.join(folder, f"GRAPH_DETAIL_SIGNED_{safe}{suffix}.png"))
    plt.close()

def _generate_leaderboard_file(participant_scores, folder):
    if not participant_scores: return
    lines = ["🏆 LEADERBOARD (Mean Abs Error in CM) 🏆"]
    for exp_name in sorted(participant_scores.keys()):
        lines.append(f"\n--- {_shorten_name(exp_name)} ---")
        scores = []
        for pid, errors in participant_scores[exp_name].items(): scores.append((np.mean(errors), pid, len(errors)))
        scores.sort(key=lambda x: x[0])
        for i, (s, p, c) in enumerate(scores[:5], 1): lines.append(f"{i}. {p}: {s:.2f} cm ({c} trials)")
    with open(os.path.join(folder, "LEADERBOARD.txt"), "w", encoding="utf-8") as f: f.write("\n".join(lines))

def _generate_statistical_report(raw_data_flat, folder):
    if not stats: return
    output_path = os.path.join(folder, "STATS_REPORT.txt")
    lines = ["📊 STATISTICAL SIGNIFICANCE REPORT (Welch's T-Test) 📊\n"]
    comparisons = [("Distance", "2. Distance Single", "3. Distance Dual"), ("Elevation", "4. Elevation Single", "5. Elevation Dual")]
    for dim, k1, k2 in comparisons:
        l1 = next((k for k in raw_data_flat if k1 in k), None)
        l2 = next((k for k in raw_data_flat if k2 in k), None)
        lines.append(f"--- {dim} ---")
        if not l1 or not l2: 
            lines.append("Missing data.\n")
            continue
        e1 = [d['abs_error'] for d in raw_data_flat[l1]]
        e2 = [d['abs_error'] for d in raw_data_flat[l2]]
        t, p = stats.ttest_ind(e1, e2, equal_var=False)
        lines.append(f"Single (N={len(e1)}): {np.mean(e1):.2f} cm")
        lines.append(f"Dual   (N={len(e2)}): {np.mean(e2):.2f} cm")
        lines.append(f"P-Value: {p:.5f} -> {'SIGNIFICANT ✅' if p<0.05 else 'NOT SIGNIFICANT ❌'}\n")
    with open(output_path, "w", encoding="utf-8") as f: f.write("\n".join(lines))
    print(f" -> Generated Stats Report: {output_path}")