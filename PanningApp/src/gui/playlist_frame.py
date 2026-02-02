import tkinter as tk
from tkinter import ttk, messagebox
import os
from .experiment_logic import EXPERIMENT_PROFILES, ExperimentSession, SymbolManager
from .data_analysis import generate_aggregated_plots

class PlaylistFrame(tk.Frame):
    def __init__(self, master, app_instance):
        super().__init__(master)
        self.app = app_instance 
        self.session = None
        self.selected_buttons = [] 
        
        self._setup_ui()

    def _setup_ui(self):
        # --- TOP: SETUP ---
        setup_frame = tk.LabelFrame(self, text="Session Setup", padx=5, pady=5)
        setup_frame.pack(fill="x", padx=10, pady=5)
        
        f1 = tk.Frame(setup_frame)
        f1.pack(fill="x", pady=2)
        tk.Label(f1, text="Participant ID:").pack(side="left")
        self.entry_pid = tk.Entry(f1, width=8)
        self.entry_pid.pack(side="left", padx=5)
        self.entry_pid.insert(0, "P01")
        
        tk.Label(f1, text="Exp:").pack(side="left", padx=10)
        self.combo_exp = ttk.Combobox(f1, values=list(EXPERIMENT_PROFILES.keys()), state="readonly", width=18)
        self.combo_exp.pack(side="left")
        self.combo_exp.current(0)

        f2 = tk.Frame(setup_frame)
        f2.pack(fill="x", pady=5)
        tk.Label(f2, text="Audio:").pack(side="left")
        audio_files = self.app.audio_files if self.app.audio_files else ["No Audio"]
        self.combo_audio = ttk.Combobox(f2, values=audio_files, state="readonly", width=15)
        self.combo_audio.pack(side="left", padx=5)
        if audio_files: self.combo_audio.current(0)
        
        # --- BUTTONS ---
        self.btn_start = tk.Button(f2, text="START SESSION", bg="#ccffcc", command=self.start_session)
        self.btn_start.pack(side="left", padx=20)
        
        # NEW BUTTON: REGENERATE GRAPHS
        self.btn_regen = tk.Button(f2, text="UPDATE GRAPHS", command=self.regenerate_graphs)
        self.btn_regen.pack(side="left", padx=5)

        # --- MID: STATUS ---
        ctrl_frame = tk.Frame(self)
        ctrl_frame.pack(fill="x", padx=10, pady=5)
        self.lbl_status = tk.Label(ctrl_frame, text="Status: Ready", font=("Arial", 14, "bold"))
        self.lbl_status.pack(side="left")
        self.btn_play = tk.Button(ctrl_frame, text="PLAY (Space)", command=self.play_trial, font=("Arial", 11, "bold"), height=2, width=15, state="disabled")
        self.btn_play.pack(side="right")

        # --- BOTTOM: SYMBOL GRID ---
        self.grid_frame = tk.Frame(self)
        self.grid_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.frame_neg = tk.LabelFrame(self.grid_frame, text="NEGATIVE (Left / Back / Down)", bg="#f0f0f0")
        self.frame_neg.pack(side="left", fill="both", expand=True, padx=2)
        self.frame_pos = tk.LabelFrame(self.grid_frame, text="POSITIVE (Right / Front / Up)", bg="#f0f0f0")
        self.frame_pos.pack(side="right", fill="both", expand=True, padx=2)
        
        btn_confirm = tk.Button(self, text="CONFIRM SELECTION (Enter)", command=self.submit_manual, bg="#ddddff", height=2)
        btn_confirm.pack(fill="x", padx=10, pady=5)
        
        self.bind_all("<space>", lambda e: self.play_trial())
        self.bind_all("<Return>", lambda e: self.submit_manual())

    def regenerate_graphs(self):
        """Manually triggers the graph generation logic."""
        try:
            generate_aggregated_plots()
            messagebox.showinfo("Success", "Graphs updated successfully!\nCheck the 'experiment_data' folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate graphs:\n{e}")

    def start_session(self):
        pid = self.entry_pid.get()
        exp_key = self.combo_exp.get()
        audio_file = self.combo_audio.get()
        
        config = EXPERIMENT_PROFILES[exp_key]
        dims = config['dimensions']
        gain = config['gain']
        
        # 1. Force App Updates
        self.app.speakers_x_entry.delete(0, tk.END)
        self.app.speakers_x_entry.insert(0, str(dims['speakers']))
        self.app.speakers_y_entry.delete(0, tk.END)
        self.app.speakers_y_entry.insert(0, str(dims['wall']))
        
        self.app.update_speaker_positions() 
        self.app.experiment_gain = gain 
        
        # 2. Retrieve calculated coordinates
        calc_coords = self.app.speaker_positions 

        self.lbl_status.config(text=f"Set to: Spk {dims['speakers']}cm | Wall {dims['wall']}cm | Gain {gain}x")

        self.app.load_audio_file(audio_file)
        
        # 3. Create Session
        self.session = ExperimentSession(exp_key, pid, audio_file, dims, speaker_coords=calc_coords)
        
        self._build_grid(self.session.config['max_distance'])
        self.btn_start.config(state="disabled")
        self.btn_play.config(state="normal")
        self._load_next_trial()

    def _build_grid(self, max_dist):
        for w in self.frame_neg.winfo_children(): w.destroy()
        for w in self.frame_pos.winfo_children(): w.destroy()
        
        configs = SymbolManager.get_button_config(max_dist)
        fg_colors = {"BLACK": "black", "RED": "#D00000", "GREEN": "#008000", "BLUE": "#0000CC"}

        def create_btn(parent, cfg, sign):
            cm_val = cfg['cm'] * sign
            
            txt = cfg['label']
            if sign == -1 and cfg['cm'] > 0:
                parts = txt.split('\n')
                if len(parts) == 2:
                    txt = f"{parts[0]}\n-{parts[1]}"

            btn = tk.Button(parent, text=txt, fg=fg_colors[cfg['color']], 
                            width=5, height=2, font=("Arial", 12, "bold"),
                            borderwidth=1, relief="raised")
            btn.config(command=lambda b=btn, v=cm_val: self.on_btn_click(b, v))
            
            idx = int(cfg['cm'] / 25)
            row = idx // 4
            col = idx % 4
            if sign == -1: col = 3 - col 
            btn.grid(row=row, column=col, padx=2, pady=2)

        for cfg in configs:
            create_btn(self.frame_pos, cfg, 1)
            if cfg['cm'] > 0:
                create_btn(self.frame_neg, cfg, -1)

    def on_btn_click(self, btn, cm_val):
        if btn in [x[0] for x in self.selected_buttons]:
            btn.config(relief="raised", borderwidth=1)
            self.selected_buttons = [x for x in self.selected_buttons if x[0] != btn]
        else:
            btn.config(relief="solid", borderwidth=4)
            self.selected_buttons.append((btn, cm_val))
            
        if len(self.selected_buttons) > 2:
            old = self.selected_buttons.pop(0)[0]
            old.config(relief="raised", borderwidth=1)

    def submit_manual(self):
        if not self.selected_buttons: return
        vals = [x[1] for x in self.selected_buttons]
        avg = sum(vals) / len(vals)
        note = "Between" if len(vals) > 1 else ""
        self.session.log_response(avg, note)
        
        for btn, _ in self.selected_buttons:
            btn.config(relief="raised", borderwidth=1)
        self.selected_buttons.clear()
        self._load_next_trial()

    def _load_next_trial(self):
        trial = self.session.next_trial()
        
        if not trial:
            self.lbl_status.config(text="DONE! Saving...", fg="blue")
            self.btn_play.config(state="disabled")
            
            success = self.session.save_session_to_disk()
            
            if success:
                try:
                    generate_aggregated_plots() 
                    msg = f"Session Saved: {self.session.filepath}\nGraphs Updated."
                except Exception as e:
                    msg = f"Saved CSV, but Graph Error: {e}"
            else:
                msg = "CRITICAL ERROR: Could not save CSV!"

            self.lbl_status.config(text="SESSION COMPLETE", fg="green")
            self.btn_start.config(state="normal") 
            messagebox.showinfo("Finished", msg)
            return
            
        total = len(self.session.playlist)
        current = self.session.current_trial_idx + 1
        self.lbl_status.config(text=f"Trial {current}/{total} | Ready")
        self.app.update_pan(trial['cm'])

    def play_trial(self):
        if self.session and self.session.get_current_trial():
            self.app.play_audio()