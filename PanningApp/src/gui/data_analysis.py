import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
from scipy.interpolate import UnivariateSpline 
from matplotlib.ticker import MultipleLocator

# --- 1. GLOBAL VISUAL SETTINGS ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 2.0,
    'figure.figsize': (8, 8),
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
    "Elevation": 160.0, 
    "Distance": 145.0   
}

# --- GLOBAL AXIS LIMITS (For Comparability) ---
# We lock both X and Y axes to +/- 75 degrees to keep scales 1:1
AXIS_LIMIT = 75.0 

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

    # --- GENERATE GRAPHS ---
    print("Generating Thesis Angular (Degree) Graphs...")
    
    # 1. Standard Error Bar Graphs (IQR)
    for exp_label, audio_groups in experiments.items():
        _plot_multi_line_experiment(exp_label, audio_groups, data_folder, use_deg=True)

    _plot_master_comparison(experiments, data_folder, use_deg=True)
    _plot_method_comparison(experiments, data_folder, "Distance", "2. Distance Single", "3. Distance Dual", use_deg=True)
    _plot_method_comparison(experiments, data_folder, "Elevation", "4. Elevation Single", "5. Elevation Dual", use_deg=True)
    
    # 2. Spline Trend Graphs
    print("Generating Reactive Spline Graphs...")
    for exp_label, audio_groups in experiments.items():
        _plot_multi_line_spline_trend(exp_label, audio_groups, data_folder)
        
    _plot_method_comparison_spline(experiments, data_folder, "Distance", "2. Distance Single", "3. Distance Dual")
    _plot_method_comparison_spline(experiments, data_folder, "Elevation", "4. Elevation Single", "5. Elevation Dual")

    # 3. Detailed & Bias
    _plot_master_absolute_error(raw_data_flat, data_folder, use_deg=True)
    _plot_master_grouped_position_error_lines(raw_data_flat, data_folder, use_deg=True)
    _plot_master_signed_error(raw_data_flat, data_folder, use_deg=True)

    for exp_label in raw_data_flat.keys():
        _plot_detailed_error_vs_position(exp_label, raw_data_flat[exp_label], data_folder, use_deg=True)

    # --- REPORTS ---
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
    if str(exp_name).lower().endswith('.wav'):
        return str(exp_name)[:-4]
    if '.' in exp_name:
        parts = exp_name.split('.')
        if len(parts) > 1: return parts[1].split('(')[0].strip()
    return exp_name

def _calculate_dynamic_limits(values, padding_percent=0.15):
    return AXIS_LIMIT

def _add_ideal_line(ax, limit):
    ax.plot([-limit, limit], [-limit, limit], color='gray', linestyle='--', linewidth=1, label='Ideal (y=x)', zorder=0)

def _add_reference_lines(ax, exp_name, use_deg, y_limit):
    radius = _get_radius_for_exp(exp_name)
    speakers, boundaries = [], []
    spk_labels, bnd_labels = [], []
    
    if "Elevation" in exp_name:
        speakers = [-92.5, 92.5]
        spk_labels = ["Floor LS", "Ceiling LS"]
        boundaries = [-127.5, 137.5]
        bnd_labels = ["Floor", "Ceiling"]
    elif "Azimuth" in exp_name:
        speakers = [-122.5, 122.5]
        spk_labels = ["Left LS", "Right LS"]
    elif "Distance" in exp_name:
        speakers = [-122.5, 122.5]
        spk_labels = ["Rear LS", "Front LS"]
        if "Dual" in exp_name:
            boundaries = [-122.5]
            bnd_labels = ["Rear Wall"]
    else:
        return

    for i, pos in enumerate(speakers):
        val = _cm_to_deg(pos, radius) if use_deg else pos
        if abs(val) > y_limit: continue
        ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=0)
        label = spk_labels[i] if i < len(spk_labels) else "LS"
        ax.text(val, y_limit*0.95, label, rotation=90, verticalalignment='top', 
                color='gray', fontsize=8, ha='right')

    for i, pos in enumerate(boundaries):
        val = _cm_to_deg(pos, radius) if use_deg else pos
        if abs(val) > y_limit: continue
        ax.axvline(x=val, color='red', linestyle=':', alpha=0.6, linewidth=1.5, zorder=0)
        label = bnd_labels[i] if i < len(bnd_labels) else "Wall"
        ax.text(val, -y_limit*0.95, label, rotation=90, verticalalignment='bottom', 
                color='red', fontsize=8, ha='right')

def _set_10deg_grid(ax):
    """Applies a 10-degree dotted grid."""
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, linestyle=':', alpha=0.6)

# ==============================================================================
# 1. ERROR BAR PLOTS (IQR)
# ==============================================================================

def _plot_multi_line_experiment(exp_name, audio_groups, folder, use_deg=False):
    if not use_deg: return
    fig, ax = plt.subplots(figsize=(7, 7))
    radius = _get_radius_for_exp(exp_name)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(audio_groups), 3)))

    for i, (audio, data) in enumerate(audio_groups.items()):
        targets = sorted(data.keys())
        medians, err_low, err_high, x_vals = [], [], [], []
        
        for t in targets:
            responses = [_cm_to_deg(r, radius) for r in data[t]]
            x_val = _cm_to_deg(t, radius)
            q1 = np.percentile(responses, 25)
            q3 = np.percentile(responses, 75)
            med = np.median(responses)
            
            x_vals.append(x_val) 
            medians.append(med)
            err_low.append(med - q1)
            err_high.append(q3 - med)
        
        ax.errorbar(x_vals, medians, yerr=[err_low, err_high], fmt='o-', 
                    label=_shorten_name(os.path.basename(audio)), 
                    color=colors[i], capsize=3, elinewidth=1.5, markersize=6, alpha=0.7)

    limit = AXIS_LIMIT
    _add_ideal_line(ax, limit)
    _add_reference_lines(ax, exp_name, True, limit)
    _set_10deg_grid(ax) 
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r'Target Angle ($^{\circ}$)')
    ax.set_ylabel(r'Perceived Angle ($^{\circ}$)')
    
    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='none', label='Dots: Median / Whiskers: IQR')
    handles.append(patch)
    
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=3, frameon=True, fontsize=9)
    
    safe = "".join([c for c in exp_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    plt.savefig(os.path.join(folder, f"GRAPH_{safe}_IQR.png"), dpi=300, bbox_inches='tight')
    plt.close()

def _plot_method_comparison(experiments, folder, dim, k1, k2, use_deg=False):
    label_single = next((k for k in experiments.keys() if k1 in k), None)
    label_dual = next((k for k in experiments.keys() if k2 in k), None)
    if not label_single or not label_dual: return
    if not use_deg: return
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    series_config = [
        (label_single, '#1f77b4', 'Single', 'o'), 
        (label_dual, '#ff7f0e', 'Dual', 's')
    ]
    
    for label, color, tag, m in series_config:
        radius = _get_radius_for_exp(label)
        merged = defaultdict(list)
        for _, data_dict in experiments[label].items():
            for t, r in data_dict.items(): merged[t].extend(r)
        
        targets = sorted(merged.keys())
        x_vals, medians, err_low, err_high = [], [], [], []
        
        for t in targets:
            responses = [_cm_to_deg(r, radius) for r in merged[t]]
            x_val = _cm_to_deg(t, radius)
            q1 = np.percentile(responses, 25)
            q3 = np.percentile(responses, 75)
            med = np.median(responses)
            
            x_vals.append(x_val) 
            medians.append(med)
            err_low.append(med - q1)
            err_high.append(q3 - med)
            
        ax.errorbar(x_vals, medians, yerr=[err_low, err_high], fmt=m, capsize=4, label=tag, color=color, alpha=0.7)

    limit = AXIS_LIMIT
    _add_ideal_line(ax, limit)
    _add_reference_lines(ax, label_single, True, limit)
    _set_10deg_grid(ax) 
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r'Target Angle ($^{\circ}$)')
    ax.set_ylabel(r'Perceived Angle ($^{\circ}$)')
    
    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='none', label='Dots: Median / Whiskers: IQR')
    handles.append(patch)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=3, frameon=True, fontsize=9)
    
    plt.savefig(os.path.join(folder, f"GRAPH_COMPARE_{dim.upper()}_IQR.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 2. SPLINE TREND GRAPHS
# ==============================================================================

def _plot_multi_line_spline_trend(exp_name, audio_groups, folder):
    fig, ax = plt.subplots(figsize=(7, 7))
    radius = _get_radius_for_exp(exp_name)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(audio_groups), 3)))

    for i, (audio, data) in enumerate(audio_groups.items()):
        targets = sorted(data.keys())
        x_vals, medians, err_low, err_high = [], [], [], []
        
        for t in targets:
            responses = [_cm_to_deg(r, radius) for r in data[t]]
            x_val = _cm_to_deg(t, radius)
            q1 = np.percentile(responses, 25)
            q3 = np.percentile(responses, 75)
            med = np.median(responses)
            
            x_vals.append(x_val)
            medians.append(med)
            err_low.append(med - q1)
            err_high.append(q3 - med)

        col = colors[i]
        label = _shorten_name(os.path.basename(audio))
        
        if len(x_vals) > 3:
            spline = UnivariateSpline(x_vals, medians, k=3, s=len(x_vals)*20) 
            x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
            y_smooth = spline(x_smooth)
            
            residuals = np.array(medians) - spline(x_vals)
            std_err = np.std(residuals)
            ax.fill_between(x_smooth, y_smooth - std_err, y_smooth + std_err, color=col, alpha=0.2, linewidth=0)
        else:
             ax.plot(x_vals, medians, '-', color=col, linewidth=2, label=f"{label} (Linear)")

        ax.errorbar(x_vals, medians, yerr=[err_low, err_high], fmt='o', color=col, alpha=0.7, markersize=4, capsize=2, label=label)

    limit = AXIS_LIMIT
    _add_ideal_line(ax, limit)
    _add_reference_lines(ax, exp_name, True, limit)
    _set_10deg_grid(ax) 
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r'Target Angle ($^{\circ}$)')
    ax.set_ylabel(r'Perceived Angle ($^{\circ}$)')
    
    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='gray', alpha=0.2, label='Curve: Spline Fit / Shade: Fit Error / Whiskers: IQR')
    handles.append(patch)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=2, frameon=True, fontsize=9)
    
    safe = "".join([c for c in exp_name if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    plt.savefig(os.path.join(folder, f"GRAPH_{safe}_Spline.png"), dpi=300, bbox_inches='tight')
    plt.close()

def _plot_method_comparison_spline(experiments, folder, dim, k1, k2):
    label_single = next((k for k in experiments.keys() if k1 in k), None)
    label_dual = next((k for k in experiments.keys() if k2 in k), None)
    if not label_single or not label_dual: return
    
    fig, ax = plt.subplots(figsize=(7, 7))
    all_vals = []
    
    series_config = [
        (label_single, '#1f77b4', 'Single', 'o'), 
        (label_dual, '#ff7f0e', 'Dual', 's')
    ]
    
    for label, color, tag, m in series_config:
        radius = _get_radius_for_exp(label)
        merged = defaultdict(list)
        for _, data_dict in experiments[label].items():
            for t, r in data_dict.items(): merged[t].extend(r)
        
        targets = sorted(merged.keys())
        x_vals, medians, err_low, err_high = [], [], [], []
        
        for t in targets:
            responses = [_cm_to_deg(r, radius) for r in merged[t]]
            x_val = _cm_to_deg(t, radius)
            q1 = np.percentile(responses, 25)
            q3 = np.percentile(responses, 75)
            med = np.median(responses)
            
            x_vals.append(x_val)
            medians.append(med)
            err_low.append(med - q1)
            err_high.append(q3 - med)
            all_vals.extend(responses)
            all_vals.append(x_val)
            
        if len(x_vals) > 3:
            spline = UnivariateSpline(x_vals, medians, k=3, s=len(x_vals)*20)
            x_smooth = np.linspace(min(x_vals), max(x_vals), 100)
            y_smooth = spline(x_smooth)
            
            residuals = np.array(medians) - spline(x_vals)
            std_err = np.std(residuals)
            ax.fill_between(x_smooth, y_smooth - std_err, y_smooth + std_err, color=color, alpha=0.2, linewidth=0)
        
        ax.errorbar(x_vals, medians, yerr=[err_low, err_high], fmt=m, color=color, alpha=0.7, markersize=5, capsize=3, label=tag)

    limit = AXIS_LIMIT
    _add_ideal_line(ax, limit)
    _add_reference_lines(ax, label_single, True, limit)
    _set_10deg_grid(ax) 
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r'Target Angle ($^{\circ}$)')
    ax.set_ylabel(r'Perceived Angle ($^{\circ}$)')
    
    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='gray', alpha=0.2, label='Curve: Spline Fit / Shade: Fit Error / Whiskers: IQR')
    handles.append(patch)
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=2, frameon=True, fontsize=9)
    
    plt.savefig(os.path.join(folder, f"GRAPH_COMPARE_{dim.upper()}_Spline.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 3. MASTER COMPARISON & DOT CHARTS
# ==============================================================================

def _plot_master_comparison(experiments, folder, use_deg=False):
    if not use_deg: return
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(experiments), 3)))

    for i, (exp_name, audio_groups) in enumerate(experiments.items()):
        merged = defaultdict(list)
        radius = _get_radius_for_exp(exp_name)
        for _, data_dict in audio_groups.items():
            for t, r in data_dict.items(): merged[t].extend(r)
        
        targets = sorted(merged.keys())
        x_vals, medians = [], []
        for t in targets:
            responses = merged[t]
            x_val = _cm_to_deg(t, radius)
            vals = [_cm_to_deg(r, radius) for r in responses]
            x_vals.append(x_val)
            medians.append(np.median(vals))
            
        ax.plot(x_vals, medians, 'o-', linewidth=2, label=_shorten_name(exp_name), color=colors[i], markersize=6)

    limit = AXIS_LIMIT
    _add_ideal_line(ax, limit)
    # NO REFERENCE LINES
    _set_10deg_grid(ax) 
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(r'Target Angle ($^{\circ}$)')
    ax.set_ylabel(r'Perceived Angle ($^{\circ}$)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
              ncol=3, frameon=True, fontsize=9)
    plt.savefig(os.path.join(folder, "GRAPH_MASTER_POSITIONS.png"), dpi=300, bbox_inches='tight')
    plt.close()

def _plot_master_absolute_error(raw_data, folder, use_deg=False):
    if not raw_data or not use_deg: return
    plt.figure(figsize=(10, 6))
    exp_names = sorted(raw_data.keys())
    means, sems, labels = [], [], []
    for exp in exp_names:
        radius = _get_radius_for_exp(exp)
        errors = [_cm_to_deg(d['abs_error'], radius) for d in raw_data[exp]]
        if not errors: continue
        means.append(np.mean(errors)) 
        sems.append(np.std(errors) / np.sqrt(len(errors)))
        labels.append(_shorten_name(exp))
        
    bars = plt.bar(np.arange(len(labels)), means, yerr=sems, capsize=6, color='#5da5da', alpha=0.9)
    plt.xticks(np.arange(len(labels)), labels, rotation=25, ha='right')
    plt.ylabel(r'Mean Absolute Error ($^{\circ}$)')
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend([bars], ["Bar: Mean Error / Whiskers: SEM"], 
               loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=1, frameon=True, fontsize=9)
    plt.savefig(os.path.join(folder, "GRAPH_MASTER_ERROR_ABSOLUTE.png"), dpi=300, bbox_inches='tight')
    plt.close()

def _plot_master_grouped_position_error_lines(raw_data_flat, folder, use_deg=False):
    if not raw_data_flat or not use_deg: return
    data_map = defaultdict(dict)
    all_positions_deg = set()
    all_experiments = sorted(raw_data_flat.keys())
    
    for exp_name in all_experiments:
        radius = _get_radius_for_exp(exp_name)
        pos_errors = defaultdict(list)
        for item in raw_data_flat[exp_name]:
            val_deg = _cm_to_deg(item['abs_error'], radius)
            tgt_deg = round(_cm_to_deg(item['target'], radius)) 
            pos_errors[tgt_deg].append(val_deg)
        for pos, errors in pos_errors.items():
            data_map[pos][exp_name] = np.median(errors)
            all_positions_deg.add(pos)
            
    sorted_positions = sorted(list(all_positions_deg))
    if not sorted_positions: return
    
    plt.figure(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_experiments)))
    
    for i, exp_name in enumerate(all_experiments):
        x_vals, y_vals = [], []
        for p in sorted_positions:
            if exp_name in data_map[p]:
                x_vals.append(p)
                y_vals.append(data_map[p][exp_name])
        # SCATTER PLOT ONLY (No lines)
        plt.plot(x_vals, y_vals, marker='o', linestyle='None', label=_shorten_name(exp_name), color=colors[i], markersize=6, alpha=0.8)
        
    plt.xlabel(r'Target Angle ($^{\circ}$)')
    plt.ylabel(r'Median Abs Error ($^{\circ}$)')
    
    ax = plt.gca()
    _set_10deg_grid(ax)
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT) 
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
               ncol=3, frameon=True, fontsize=9)
    plt.savefig(os.path.join(folder, "GRAPH_MASTER_DETAILED_LINES.png"), dpi=300, bbox_inches='tight')
    plt.close()

def _plot_master_signed_error(raw_data, folder, use_deg=False):
    if not raw_data or not use_deg: return
    plt.figure(figsize=(10, 6))
    exp_names = sorted(raw_data.keys())
    data, labels = [], []
    for exp in exp_names:
        radius = _get_radius_for_exp(exp)
        errors = [_cm_to_deg(d['signed_error'], radius) for d in raw_data[exp]]
        if not errors: continue
        data.append(errors)
        labels.append(_shorten_name(exp))
        
    plt.axhline(0, color='black')
    bp = plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="#90EE90", color="black"))
    plt.xticks(rotation=90) 
    plt.ylabel(r'Signed Error ($^{\circ}$)')
    plt.grid(axis='y', linestyle=':')
    plt.legend([bp["boxes"][0]], ["Box: IQR / Line: Median"], 
               loc='upper center', bbox_to_anchor=(0.5, -0.25), 
               ncol=1, frameon=True, fontsize=9)
    plt.savefig(os.path.join(folder, "GRAPH_MASTER_BIAS.png"), dpi=300, bbox_inches='tight')
    plt.close()

def _plot_detailed_error_vs_position(exp_name, data_list, folder, use_deg=False):
    if not use_deg: return
    grouped = defaultdict(list)
    radius = _get_radius_for_exp(exp_name)
    for item in data_list:
        val = _cm_to_deg(item['signed_error'], radius)
        tgt = round(_cm_to_deg(item['target'], radius), 1)
        grouped[tgt].append(val)
        
    targets = sorted(grouped.keys())
    if not targets: return
    
    plt.figure(figsize=(9, 5))
    plt.axhline(0, color='black', linewidth=1)
    
    # BOXPLOT
    bp = plt.boxplot([grouped[t] for t in targets], positions=targets, widths=3, showfliers=False,
                boxprops=dict(color='black'), medianprops=dict(color='red'))
    plt.xlabel(r'Target Angle ($^{\circ}$)')
    plt.ylabel(r'Signed Error ($^{\circ}$)')
    
    ax = plt.gca()
    _set_10deg_grid(ax)
    
    # FIXED LIMITS
    ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT) # 1:1 Scale
    
    plt.legend([bp["boxes"][0]], ["Box: IQR / Line: Median"], 
               loc='upper center', bbox_to_anchor=(0.5, -0.2), 
               ncol=1, frameon=True, fontsize=9)
    
    _add_reference_lines(plt.gca(), exp_name, True, AXIS_LIMIT)
    
    safe = "".join([c for c in _shorten_name(exp_name) if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"GRAPH_DETAIL_BIAS_{safe}.png"), dpi=300, bbox_inches='tight')
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