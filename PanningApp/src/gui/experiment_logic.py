import csv
import random
import os
import sys
from datetime import datetime

# ==============================================================================
# 1. EXPERIMENT PROFILES
# ==============================================================================
EXPERIMENT_PROFILES = {
    "EXP_1_AZIMUTH": {
        "label": "1. Azimuth (Left / Right)",
        "max_distance": 300.0,
        "directions": ("LEFT", "RIGHT"),
        "dimensions": {"speakers": 245.0, "wall": 212.0},
        "gain": 0.5 # Dual Mono
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
# 3. SESSION MANAGER (BUFFERED SAVE)
# ==============================================================================
class ExperimentSession:
    def __init__(self, experiment_key, participant_id, audio_filename, dimensions, speaker_coords=None):
        self.config = EXPERIMENT_PROFILES[experiment_key]
        self.participant_id = participant_id
        self.audio_filename = audio_filename
        self.dimensions = dimensions 
        self.speaker_coords = speaker_coords # NEW: Store actual coordinates
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # --- PATH FIX ---
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        self.data_dir = os.path.join(project_root, "experiment_data")
        
        if not os.path.exists(self.data_dir):
            try: os.makedirs(self.data_dir)
            except OSError: self.data_dir = "experiment_data"
        
        # Consolidated file per participant
        self.filepath = os.path.join(self.data_dir, f"{participant_id}.csv")
        
        # Buffer: Data stays here until 'save_session_to_disk' is called
        self.data_buffer = []
        
        self.playlist = self._generate_playlist()
        self.current_trial_idx = -1

    def _generate_playlist(self):
        max_d = int(self.config['max_distance'])
        neg_label, pos_label = self.config['directions']
        
        possible_positions = list(range(25, max_d + 1, 25))
        neg_trials = [{"cm": -p, "side": neg_label} for p in possible_positions]
        pos_trials = [{"cm": p, "side": pos_label} for p in possible_positions]
        
        random.shuffle(neg_trials)
        random.shuffle(pos_trials)
        
        playlist = []
        for i in range(min(len(neg_trials), len(pos_trials))):
            playlist.append(neg_trials[i])
            playlist.append(pos_trials[i])
            
        center_trial = {"cm": 0, "side": "CENTER"}
        playlist.insert(random.randint(0, len(playlist)//2), center_trial)
        
        # --- DETAILED TERMINAL REPORT ---
        print("\n" + "="*60)
        print(f" EXPERIMENT STARTED: {self.config['label']}")
        print(f" PARTICIPANT: {self.participant_id}")
        print("-" * 60)
        print(f" CONFIGURATIONS:")
        print(f"   > Audio File:   {self.audio_filename}")
        print(f"   > Setup Dims:   Speakers={self.dimensions['speakers']}cm | Wall={self.dimensions['wall']}cm")
        print(f"   > Gain Factor:  {self.config['gain']}x")
        if self.speaker_coords:
            # Print cleanly formatted numpy arrays
            print(f"   > Calculated Coords:")
            print(f"       Left Speaker:  {self.speaker_coords[0]}")
            print(f"       Right Speaker: {self.speaker_coords[1]}")
        print("-" * 60)
        print(f" PLAYLIST ({len(playlist)} Trials):")
        for i, t in enumerate(playlist):
            print(f"   {i+1:>2}. {t['side']:<8} {t['cm']:>4} cm")
        print("="*60 + "\n")
        # --------------------------------
        
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
        """Store response in RAM. Do NOT write to disk yet."""
        trial = self.get_current_trial()
        if not trial: return
        target = trial['cm']
        error = abs(response_cm - target)
        
        row = [self.current_trial_idx + 1, trial['side'], target, response_cm, f"{error:.2f}", notes]
        self.data_buffer.append(row)

    def save_session_to_disk(self):
        """Called ONLY when experiment completes successfully."""
        file_exists = os.path.exists(self.filepath)
        
        try:
            with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Separation between sessions
                if file_exists:
                    writer.writerow([])
                    writer.writerow([])
                    writer.writerow(["--- NEW SESSION ---"])
                
                # Session Header
                writer.writerow(["Experiment", self.config['label']])
                writer.writerow(["Participant", self.participant_id])
                writer.writerow(["Audio", self.audio_filename])
                writer.writerow(["Gain Factor", self.config['gain']])
                writer.writerow(["Date", self.start_time])
                writer.writerow(["Speaker_Dist_CM", self.dimensions.get('speakers', 'N/A')])
                writer.writerow(["Wall_Dist_CM", self.dimensions.get('wall', 'N/A')])
                
                if self.speaker_coords:
                    writer.writerow(["Calc_Coords_Left", self.speaker_coords[0]])
                    writer.writerow(["Calc_Coords_Right", self.speaker_coords[1]])

                writer.writerow([])
                writer.writerow(["Trial_ID", "Target_Side", "Target_CM", "Response_CM", "Error_CM", "Notes"])
                
                # Flush Data Buffer
                writer.writerows(self.data_buffer)
                
            print(f" -> SUCCESS: Session saved to {self.filepath}")
            return True
        except Exception as e:
            print(f" -> ERROR SAVING CSV: {e}")
            return False