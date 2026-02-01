import numpy as np

# ==============================================================================
#   USER CONFIGURATION
# ==============================================================================
# Room & Speaker Dimensions (cm)
DIST_FLOOR_TO_CEILING = 265.0
DIST_CEIL_TO_TOP_TWEETER = 45.0
DIST_FLOOR_TO_BOT_TWEETER = 35.0

# Calculate exact Tweeter Positions relative to Ear/Center
# (Assuming Ear is exactly mid-height of the arcs, as per your config below)
HEIGHT_EAR_TO_CEIL = 137.5
HEIGHT_EAR_TO_FLOOR = 127.5

POS_TOP_TWEETER_FROM_CENTER = HEIGHT_EAR_TO_CEIL - DIST_CEIL_TO_TOP_TWEETER  # 137.5 - 45 = 92.5
POS_BOT_TWEETER_FROM_CENTER = HEIGHT_EAR_TO_FLOOR - DIST_FLOOR_TO_BOT_TWEETER # 127.5 - 35 = 92.5

# Panning Settings
MAX_PAN_LENGTH = 700.0       # Total Virtual Arc Length
MAX_HALF_SPAN = MAX_PAN_LENGTH / 2.0  # Distance from Center to Max Edge (350.0)
GRID_STEP = 25.0             # 25cm increments

CONFIG_1 = {
    "NAME":             "SETUP 1: On-Axis (Standard)",
    "DIST_EAR_TO_WALL": 215.0,
}

CONFIG_2 = {
    "NAME":             "SETUP 2: Dual Mono (Back position)",
    "DIST_EAR_TO_WALL": 150.0,
}

# ==============================================================================
#   LOGIC
# ==============================================================================

def calculate_markers_from_center(cfg, direction="UP"):
    
    if direction == "UP":
        boundary_limit = HEIGHT_EAR_TO_CEIL # 132.5
        tweeter_pos = POS_TOP_TWEETER_FROM_CENTER # 87.5
        boundary_name = "CEILING"
        print(f"\n--- {cfg['NAME']} : UP (Ceiling) ---")
        print(f"Top Tweeter is at {tweeter_pos:.1f}cm from Center.")
        
    else: # DOWN
        boundary_limit = HEIGHT_EAR_TO_FLOOR # 132.5
        tweeter_pos = POS_BOT_TWEETER_FROM_CENTER # 97.5
        boundary_name = "FLOOR"
        print(f"\n--- {cfg['NAME']} : DOWN (Floor) ---")
        print(f"Bottom Tweeter is at {tweeter_pos:.1f}cm from Center.")

    print("-" * 95)
    print(f"{'Pan Position':<20} | {'Status':<20} | {'INSTRUCTION'}")
    print(f"{'(cm from Center)':<20} | {'(Speaker/Room)':<20} | {'(Where to put tape)'}")
    print("-" * 95)

    wall_dist = cfg["DIST_EAR_TO_WALL"]
    
    # Iterate from 0 to 175cm (Half of the 350cm arc)
    current_pos_from_center = 0.0
    
    while current_pos_from_center <= MAX_HALF_SPAN + 0.1:
        
        # 1. Determine Context (For user reference)
        if current_pos_from_center < tweeter_pos:
            status = "Inside Speaker"
        elif current_pos_from_center < boundary_limit:
            status = "Outside Spk (Wall)"
        else:
            status = f"Outside Room ({boundary_name})"

        # 2. Calculate Projection
        
        # CASE A: Still on the Front Wall
        if current_pos_from_center <= boundary_limit:
            surface = "FRONT WALL"
            instruction = f"Mark exactly at {current_pos_from_center:.1f}cm (On the Wall)"
            
        # CASE B: Hit the Ceiling/Floor -> Project onto surface
        else:
            surface = boundary_name
            
            # Geometry: Similar Triangles
            # We are looking at a point 'current_pos' high. 
            # The wall cuts us off at 'boundary_limit' high.
            
            # Ratio = Room_Boundary_Height / Virtual_Target_Height
            ratio = boundary_limit / current_pos_from_center
            
            # How far from the head does the line hit the ceiling?
            intersect_dist_from_head = wall_dist * ratio
            
            # Convert to "Distance OUT from Wall" for easy measuring
            marker_from_wall = wall_dist - intersect_dist_from_head
            
            instruction = f"Measure {marker_from_wall:.1f}cm OUT from Wall (on {boundary_name})"

        print(f"{current_pos_from_center:>6.1f} cm{'':<11} | {status:<20} | {instruction}")
        
        current_pos_from_center += GRID_STEP
    print("\n")

# ==============================================================================
#   RUN
# ==============================================================================

for config in [CONFIG_1, CONFIG_2]:
    print("\n" + "="*50)
    print(f"   {config['NAME']}")
    print("="*50)
    calculate_markers_from_center(config, "UP")
    calculate_markers_from_center(config, "DOWN")