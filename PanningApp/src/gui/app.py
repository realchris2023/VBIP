import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import Tk, Frame, Scale, HORIZONTAL, StringVar, OptionMenu, Button, Entry, Label, Toplevel
from components.play_button import PlayButton
from audio.vbap import calculate_gains
from gui.playlist_frame import PlaylistFrame
from gui.plot import plot_audio_channels, plot_speaker_and_source_positions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class AudioPanningApp:
    def __init__(self, master):
        self.pan_range_scale_limit = 10
        self.master = master
        self.frame = Frame(master)
        self.frame.pack()
        
        # Default Settings
        self.default_speaker_x_distance = 250
        self.default_speaker_y_distance = 216.5
        self.speaker_positions = [
            np.array([-125, 216.5]),
            np.array([125, 216.5])
        ]
        self.virtual_source_position = np.array([0, self.speaker_positions[0][1]])
        
        # Audio State
        self.processed_buffer = None
        self.experiment_gain = 1.0  # <--- NEW: Default gain factor (1.0 = 0dB)

        # UI Components
        self.play_button = PlayButton(self.frame, self.play_audio)
        self.play_button.pack(side='left', padx=5, pady=5)
        self.stop_button = Button(self.frame, text="Stop", command=self.stop_audio)
        self.stop_button.pack(side='left', padx=5, pady=5)
        self.pan_knob = Scale(self.frame, from_= self.pan_range_scale_limit*(self.speaker_positions[0][0]), to= self.pan_range_scale_limit*self.speaker_positions[1][0], orient=HORIZONTAL, resolution=25, command=self.update_pan, length=400, sliderlength=30)
        self.pan_knob.set(0.0)
        self.pan_knob.pack()

        # FIX FOR PYINSTALLER
        if getattr(sys, 'frozen', False):
        # Running as compiled app
            base_path = sys._MEIPASS
            self.audio_directory = os.path.join(base_path, "src", "audio", "audio_files")
        else:
        # Running as script
            self.audio_directory = os.path.join(os.path.dirname(__file__), "..", "audio/audio_files")
            
        self._check_audio_directory()
        self.audio_files = self._get_audio_files()
        self.selected_audio = StringVar()
        if self.audio_files:
            self.selected_audio.set(self.audio_files[0])
            self.audio_menu = OptionMenu(self.frame, self.selected_audio, *self.audio_files, command=self.load_audio_file)
            self.audio_file = os.path.join(self.audio_directory, self.selected_audio.get())
            try: self.load_audio_file(self.audio_file)
            except Exception as e: print(e)
            self.audio_menu.pack()
        else:
            self.selected_audio.set("")
            self.audio_menu = OptionMenu(self.frame, self.selected_audio, "")
            self.audio_menu.pack()

        self.create_speaker_input_fields()
        self.stream = None
        self.update_pan(self.pan_knob.get())
        self.playback_index = 0
        
        self.plot_button = Button(self.frame, text="Plot Waveform", command=self.plot_current_audio)
        self.plot_button.pack()
        self.scatter_plot_button = Button(self.frame, text="Plot Speaker/Source", command=self.plot_scatter_positions)
        self.scatter_plot_button.pack()
        Label(self.frame, text="Save Audio - Postfix:").pack()
        self.save_postfix_entry = Entry(self.frame)
        self.save_postfix_entry.pack()
        self.save_button = Button(self.frame, text="Save Audio", command=self.save_audio)
        self.save_button.pack()

        Label(self.frame, text="-----------------").pack(pady=10)
        self.btn_exp = Button(self.frame, text=">> OPEN EXPERIMENT MODE <<", bg="lightblue", command=self.open_experiment_window)
        self.btn_exp.pack(pady=5, fill="x")

    def open_experiment_window(self):
        win = Toplevel(self.master)
        win.title("Experiment Controller")
        win.geometry("900x600")
        exp_interface = PlaylistFrame(win, self)
        exp_interface.pack(fill="both", expand=True)

    def create_speaker_input_fields(self):
        Label(self.frame, text="Horizontal distance between speakers(cm):").pack()
        self.speakers_x_entry = Entry(self.frame)
        self.speakers_x_entry.pack()
        self.speakers_x_entry.insert(0, "250")
        Label(self.frame, text="Distance to wall (cm):").pack()
        self.speakers_y_entry = Entry(self.frame)
        self.speakers_y_entry.pack()
        self.speakers_y_entry.insert(0, "216.5")
        update_button = Button(self.frame, text="Update Speaker Positions", command=self.update_speaker_positions)
        update_button.pack()

    def update_speaker_positions(self):
        try:
            speakers_x = float(self.speakers_x_entry.get())
            left_x = float(np.negative(speakers_x / 2))
            right_x = speakers_x / 2
            y = float(self.speakers_y_entry.get())
            self.speaker_positions = [np.array([left_x, y]), np.array([right_x, y])]
            self.pan_knob.config(from_= self.pan_range_scale_limit * (self.speaker_positions[0][0]), to=self.pan_range_scale_limit * (self.speaker_positions[1][0]))
            self.virtual_source_position = np.array([0, y])
            print(f"Updated speaker positions: {self.speaker_positions}")
        except ValueError: print("Invalid input.")

    def _check_audio_directory(self):
        if not os.path.exists(self.audio_directory): print(f"Audio directory not found: {self.audio_directory}")
    def _get_audio_files(self): return [f for f in os.listdir(self.audio_directory) if f.endswith(".wav")]

    def load_audio_file(self, filename):
        if os.path.isabs(filename): path = filename
        else: path = os.path.join(self.audio_directory, filename)
        self.audio_file = path
        try: self.audio_samples, self.sample_rate = sf.read(self.audio_file, dtype='float32')
        except Exception as e:
            print(f"Error reading audio file: {e}")
            self.audio_samples = np.zeros(1, dtype='float32')
            self.sample_rate = 44100
            return
        if hasattr(self.audio_samples, 'ndim') and self.audio_samples.ndim == 2:
            self.audio_samples = np.mean(self.audio_samples, axis=1)

    def update_pan(self, value): 
        pan_value = float(value)
        self.virtual_source_position = self.get_virtual_source_position(pan_value, self.speaker_positions[0][1])
        gains = calculate_gains(self.speaker_positions[0], self.speaker_positions[1], self.virtual_source_position)
        self.left_gain = gains[0]
        self.right_gain = gains[1]
        print(f"Pan: {pan_value:.2f}, Gains L/R: {self.left_gain:.2f}/{self.right_gain:.2f}")
    
    def get_virtual_source_position(self, pan_value, y_distance):
        return np.array([pan_value, y_distance])

    def play_audio(self):
        self._safe_close_stream()
        self.playback_index = 0 
        if not hasattr(self, 'sample_rate') or not hasattr(self, 'audio_samples'):
            print("No audio loaded to play.")
            return

        # --- APPLY VBAP GAIN + EXPERIMENT GAIN ---
        # Dual Mono experiments will have experiment_gain = 0.5 (-6dB)
        left_ch = self.audio_samples * self.left_gain * self.experiment_gain
        right_ch = self.audio_samples * self.right_gain * self.experiment_gain
        
        self.processed_buffer = np.column_stack((left_ch, right_ch))
        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2, dtype='float32', callback=self.audio_callback)
        self.stream.start()

    def audio_callback(self, outdata, frames, time, status):
        if status: print(f"Stream error: {status}")
        if self.playback_index >= len(self.processed_buffer):
            outdata.fill(0); return
        end_idx = min(self.playback_index + frames, len(self.processed_buffer))
        chunk = self.processed_buffer[self.playback_index:end_idx]
        if len(chunk) < frames: chunk = np.pad(chunk, ((0, frames - len(chunk)), (0, 0)), 'constant')
        outdata[:] = chunk
        self.playback_index += frames

    def stop_audio(self):
        self._safe_close_stream()
        self.playback_index = 0
    def _safe_close_stream(self):
        if getattr(self, 'stream', None) is not None:
            try: self.stream.stop(); self.stream.close()
            except Exception: pass
            finally: self.stream = None
    
    def plot_current_audio(self):
        start_idx = self.playback_index
        chunk_size = min(1000, len(self.audio_samples) - start_idx)
        audio_chunk = self.audio_samples[start_idx:start_idx + chunk_size]
        plot_audio_channels(audio_chunk, self.left_gain, self.right_gain, float(self.pan_knob.get()))
        
    def plot_scatter_positions(self):
        plot_speaker_and_source_positions(self.speaker_positions, self.virtual_source_position, self.pan_range_scale_limit)
    
    def save_audio(self):
        postfix = int(self.virtual_source_position[0])
        if postfix or postfix == 0:
            save_filename = os.path.splitext(self.audio_file)[0] + f"_{postfix}.wav"
            left_channel = self.audio_samples * self.left_gain * self.experiment_gain
            right_channel = self.audio_samples * self.right_gain * self.experiment_gain
            stereo_audio = np.column_stack((left_channel, right_channel))
            sf.write(save_filename, stereo_audio, self.sample_rate)
            print(f"Audio saved as {save_filename}")