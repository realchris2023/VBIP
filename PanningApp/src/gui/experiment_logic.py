import csv
import random
import os
from datetime import datetime

# ==============================================================================
# 1. EXPERIMENT PROFILES (Hardcoded Dimensions & Gain)
# ==============================================================================
EXPERIMENT_PROFILES = {
    "EXP_1_AZIMUTH": {
        "label": "1. Azimuth (Left / Right)",
        "max_distance": 300.0,
        "directions": ("LEFT", "RIGHT"),
        "dimensions": {"speakers": 245.0, "wall": 212.0},
        "gain": 1.0 
    },
    "EXP_2_DIST_SINGLE": {
        "label": "2. Distance Single (Back / Front)",
        "max_distance": 300.0,
        "directions": ("BACK", "FRONT"),
        "dimensions": {"speakers": 245.0, "wall": 150.0},
        "gain": 1.0
    },
    "EXP_3_DIST_DUAL": {
        "label": "3. Distance Dual (Back / Front)",
        "max_distance": 300.0,
        "directions": ("BACK", "FRONT"),
        "dimensions": {"speakers": 245.0, "wall": 150.0},
        "gain": 0.5 
    },
    "EXP_4_ELEV_SINGLE": {
        "label": "4. Elevation Single (Down / Up)",
        "max_distance": 300.0,
        "directions": ("DOWN", "UP"),
        "dimensions": {"speakers": 185.0, "wall": 160.0},
        "gain": 1.0
    },
    "EXP_5_ELEV_DUAL": {
        "label": "5. Elevation Dual (Down / Up)",
        "max_distance": 300.0,
        "directions": ("DOWN", "UP"),
        "dimensions": {"speakers": 185.0, "wall": 160.0},
        "gain": 0.5
    }
}

# ==============================================================================
# 2. SYMBOL DECODER
# ==============================================================================
class SymbolManager:
    COLORS = ["BLACK", "RED", "GREEN", "BLUE"]
    SHAPES = ["LINE", "SQUARE", "TRIANGLE", "CIRCLE"]

    @staticmethod
    def get_button_config(max_dist):
        buttons = []
        current_cm = 0
        while current_cm <= max_dist:
            color_idx = min(int(current_cm // 100), len(SymbolManager.COLORS) - 1)
            shape_idx = int((current_cm % 100) / 25)
            
            shape_name = SymbolManager.SHAPES[shape_idx]
            color_name = SymbolManager.COLORS[color_idx]
            
            symbols = {"LINE": "▬", "SQUARE": "■", "TRIANGLE": "▲", "CIRCLE": "●"}
            label = f"{symbols[shape_name]}\n{int(current_cm)}"
            
            buttons.append({
                "label": label, "color": color_name, "cm": current_cm, "shape": shape_name
            })
            current_cm += 25
        return buttons

# ==============================================================================
# 3. SESSION MANAGER
# ==============================================================================
class ExperimentSession:
    def __init__(self, experiment_key, participant_id, audio_filename, dimensions):
        self.config = EXPERIMENT_PROFILES[experiment_key]
        self.participant_id = participant_id
        self.audio_filename = audio_filename
        self.dimensions = dimensions 
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if not os.path.exists("experiment_data"):
            os.makedirs("experiment_data")
            
        self.filepath = f"experiment_data/{participant_id}_{experiment_key}_{self.start_time}.csv"
        self._init_csv()
        
        self.playlist = self._generate_playlist()
        self.current_trial_idx = -1

    def _init_csv(self):
        with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Experiment", self.config['label']])
            writer.writerow(["Participant", self.participant_id])
            writer.writerow(["Audio", self.audio_filename])
            writer.writerow(["Gain Factor", self.config['gain']])
            writer.writerow(["Date", self.start_time])
            writer.writerow(["Speaker_Dist_CM", self.dimensions.get('speakers', 'N/A')])
            writer.writerow(["Wall_Dist_CM", self.dimensions.get('wall', 'N/A')])
            writer.writerow([])
            writer.writerow(["Trial_ID", "Target_Side", "Target_CM", "Response_CM", "Error_CM", "Notes"])

    def _generate_playlist(self):
        max_d = int(self.config['max_distance'])
        neg_label, pos_label = self.config['directions']
        
        # Grid steps: 25, 50 ... max
        possible_positions = list(range(25, max_d + 1, 25))
        
        neg_trials = [{"cm": -p, "side": neg_label} for p in possible_positions]
        pos_trials = [{"cm": p, "side": pos_label} for p in possible_positions]
        
        random.shuffle(neg_trials)
        random.shuffle(pos_trials)
        
        playlist = []
        # Strict Alternation
        for i in range(min(len(neg_trials), len(pos_trials))):
            playlist.append(neg_trials[i])
            playlist.append(pos_trials[i])
            
        center_trial = {"cm": 0, "side": "CENTER"}
        playlist.insert(random.randint(0, len(playlist)//2), center_trial)

        # --- TERMINAL OUTPUT ---
        print("\n" + "="*50)
        print(f" EXPERIMENT: {self.config['label']}")
        print(f" DIMS: Spk={self.dimensions['speakers']}cm | Wall={self.dimensions['wall']}cm")
        print(f" GAIN: {self.config['gain']}x")
        print("-" * 50)
        print(f" FULL PLAYLIST ({len(playlist)} Trials):")
        print("-" * 50)
        for i, t in enumerate(playlist):
            # Print clearly: "1. LEFT -125 cm"
            print(f" {i+1:>2}. {t['side']:<8} {t['cm']:>5} cm")
        print("="*50 + "\n")
        # -----------------------
        
        return playlist

    def get_current_trial(self):
        if 0 <= self.current_trial_idx < len(self.playlist):
            return self.playlist[self.current_trial_idx]
        return None

    def next_trial(self):
        self.current_trial_idx += 1
        if self.current_trial_idx >= len(self.playlist):
            return None
        return self.get_current_trial()

    def log_response(self, response_cm, notes=""):
        trial = self.get_current_trial()
        if not trial: return
        target = trial['cm']
        error = abs(response_cm - target)
        
        row = [self.current_trial_idx + 1, trial['side'], target, response_cm, f"{error:.2f}", notes]
        
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)