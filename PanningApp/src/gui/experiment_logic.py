import csv
import random
import os
from datetime import datetime

# ==============================================================================
# 1. EXPERIMENT PROFILES
# ==============================================================================
EXPERIMENT_PROFILES = {
    "EXP_1_AZIMUTH": {
        "label": "1. Azimuth (Left / Right)",
        "max_distance": 350.0, # Will generate up to this point
        "directions": ("LEFT", "RIGHT"), # (Negative, Positive)
    },
    "EXP_2_DIST_SINGLE": {
        "label": "2. Distance Single (Back / Front)",
        "max_distance": 350.0,
        "directions": ("BACK", "FRONT"),
    },
    "EXP_3_DIST_DUAL": {
        "label": "3. Distance Dual (Back / Front)",
        "max_distance": 300.0,
        "directions": ("BACK", "FRONT"),
    },
    "EXP_4_ELEV_SINGLE": {
        "label": "4. Elevation Single (Down / Up)",
        "max_distance": 350.0,
        "directions": ("DOWN", "UP"),
    },
    "EXP_5_ELEV_DUAL": {
        "label": "5. Elevation Dual (Down / Up)",
        "max_distance": 300.0,
        "directions": ("DOWN", "UP"),
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
        """Generates the list of buttons needed for the UI."""
        buttons = []
        current_cm = 0
        
        while current_cm <= max_dist:
            # Color Group (Every 100cm)
            color_idx = int(current_cm // 100)
            if color_idx >= len(SymbolManager.COLORS): 
                color_idx = len(SymbolManager.COLORS) - 1
            color_name = SymbolManager.COLORS[color_idx]
            
            # Shape (Every 25cm)
            remainder = current_cm % 100
            shape_idx = int(remainder / 25)
            shape_name = SymbolManager.SHAPES[shape_idx]
            
            # Visual Label
            if shape_name == "LINE":       symbol = "▬"
            elif shape_name == "SQUARE":   symbol = "■"
            elif shape_name == "TRIANGLE": symbol = "▲"
            elif shape_name == "CIRCLE":   symbol = "●"
            
            label = f"{symbol}\n{int(current_cm)}"
            
            buttons.append({
                "label": label,
                "color": color_name,
                "cm": current_cm,
                "shape": shape_name
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
        self.dimensions = dimensions # Dict: {'speakers': X, 'wall': Y}
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
            # Header Block
            writer.writerow(["Experiment", self.config['label']])
            writer.writerow(["Participant", self.participant_id])
            writer.writerow(["Audio", self.audio_filename])
            writer.writerow(["Date", self.start_time])
            # Dimensions
            writer.writerow(["Speaker_Dist_CM", self.dimensions.get('speakers', 'N/A')])
            writer.writerow(["Wall_Dist_CM", self.dimensions.get('wall', 'N/A')])
            writer.writerow([])
            # Data Columns
            writer.writerow([
                "Trial_ID", "Target_Side", "Target_CM", 
                "Response_CM", "Error_CM", "Notes"
            ])

    def _generate_playlist(self):
        """
        Generates a 25cm grid playlist with strict side alternation.
        """
        max_d = int(self.config['max_distance'])
        neg_label, pos_label = self.config['directions']
        
        # 1. Create the Grid (0, 25, 50... up to Max)
        # Note: If you want to trim the last 50cm, reduce 'max_distance' in config above.
        possible_positions = list(range(25, max_d + 1, 25))
        
        # 2. Create Left/Right lists
        neg_trials = [{"cm": -p, "side": neg_label} for p in possible_positions]
        pos_trials = [{"cm": p, "side": pos_label} for p in possible_positions]
        
        # 3. Shuffle them independently
        random.shuffle(neg_trials)
        random.shuffle(pos_trials)
        
        # 4. Interleave them (L, R, L, R...)
        playlist = []
        # Determine strict alternation count
        min_len = min(len(neg_trials), len(pos_trials))
        
        for i in range(min_len):
            playlist.append(neg_trials[i])
            playlist.append(pos_trials[i])
            
        # Add the Center (0) somewhere random or at start?
        # Let's add Center randomly into the first half to calibrate user
        center_trial = {"cm": 0, "side": "CENTER"}
        insert_idx = random.randint(0, len(playlist)//2)
        playlist.insert(insert_idx, center_trial)
        
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
        
        row = [
            self.current_trial_idx + 1,
            trial['side'],
            target,
            response_cm,
            f"{error:.2f}",
            notes
        ]
        
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)