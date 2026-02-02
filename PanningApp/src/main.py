import sys
import os
from tkinter import Tk

# --- PATH FIX ---
# 1. Get the absolute path to 'src'
src_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the absolute path to 'src/gui' (where components lives)
gui_dir = os.path.join(src_dir, 'gui')

# 3. Add BOTH to Python's search path
#    - 'src' lets us find: gui.app, audio.vbap
#    - 'gui' lets us find: components
sys.path.insert(0, src_dir)
sys.path.insert(0, gui_dir)
# ----------------

try:
    from gui.app import AudioPanningApp
except ImportError as e:
    print(f"Startup Error: {e}")
    print(f"Looking in: {sys.path[:2]}")
    sys.exit(1)

def center_window(root, width=800, height=600):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - height / 2)
    position_right = int(screen_width / 2 - width / 2)
    root.geometry(f"{width}x{height}+{position_right}+{position_top}")

if __name__ == "__main__":
    root = Tk()
    root.title("VBAP Panning Experiment")
    center_window(root)
    root.configure(bg='#f0f0f0') 
    
    app = AudioPanningApp(master=root)
    root.mainloop()