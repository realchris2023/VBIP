import numpy as np

# ==========================================
#   USER CONFIGURATION (EDIT THIS SECTION)
# ==========================================

# 1. LISTENER POSITION
# Distances from the Subject's Ear to the physical boundaries
DIST_TO_FRONT_WALL = 2.00   # Meters
DIST_TO_CEILING    = 1.50   # Meters (Room Height - Ear Height)
DIST_TO_FLOOR      = 1.20   # Meters (Ear Height)
DIST_TO_SIDE_WALL  = 2.50   # Meters (Distance to Left/Right wall)

# 2. EXPERIMENT SETTINGS
GRID_STEP_SIZE     = 0.25   # Step size for markers (e.g., every 25cm)
MAX_EXTENSION      = 3.00   # How far outside the room to calculate

# ==========================================
#      CALCULATION LOGIC (DO NOT EDIT)
# ==========================================

def calculate_projection(boundary_dist, perpendicular_dist, direction_name, surface_name):
    """
    Calculates the projection of a virtual source onto a perpendicular surface.
    
    Args:
        boundary_dist (float): Distance from Ear to the boundary we are extending PAST (e.g., Ceiling height)
        perpendicular_dist (float): Distance from Ear to the wall we are projecting AGAINST (e.g., Front Wall)
        direction_name (str): Label for the output (e.g., "UP (Ceiling)")
        surface_name (str): Where the tape goes (e.g., "Ceiling")
    """
    print(f"\n{'='*50}")
    print(f"  DIMENSION: {direction_name}")
    print(f"  Placement Surface: {surface_name}")
    print(f"  Reference Corner: 0.00m starts at the wall/corner")
    print(f"{'='*50}")
    print(f"{'Virtual Ext.':<15} | {'Calculation (Ratio)':<25} | {'MARKER POS':<15}")
    print(f"{'(Meters)':<15} | {'(Head vs Virtual)':<25} | {'(From Wall)':<15}")
    print("-" * 65)

    # 0.00 is always the corner
    print(f"{'+ 0.00 m':<15} | {'(Corner)':<25} | {'0.00 m':<15}")

    current_ext = GRID_STEP_SIZE
    while current_ext <= MAX_EXTENSION:
        
        # 1. Total Virtual Distance/Height from Ear
        total_virtual_dist = boundary_dist + current_ext
        
        # 2. Ratio (The triangle scaling factor)
        # We want to scale the 'Perpendicular Distance' down to fit the room
        scaling_ratio = boundary_dist / total_virtual_dist
        
        # 3. Distance from Head to the Intersection Point
        intersect_dist_from_head = perpendicular_dist * scaling_ratio
        
        # 4. Marker Position (Distance from the Wall/Corner)
        # We subtract from the total wall distance to find distance FROM the wall
        marker_pos = perpendicular_dist - intersect_dist_from_head
        
        print(f"+ {current_ext:.2f} m{'':<8} | {boundary_dist:.2f}/{total_virtual_dist:.2f} = {scaling_ratio:.3f}{'':<9} | {marker_pos:.2f} m")
        
        current_ext += GRID_STEP_SIZE
    print("-" * 65)

# ==========================================
#        RUN CALCULATIONS
# ==========================================

# 1. ELEVATION UP (Projecting onto Ceiling)
# We extend PAST the Ceiling height, looking AT the Front Wall
calculate_projection(
    boundary_dist=DIST_TO_CEILING, 
    perpendicular_dist=DIST_TO_FRONT_WALL, 
    direction_name="UP (Elevation)", 
    surface_name="CEILING (Distance from Front Wall)"
)

# 2. ELEVATION DOWN (Projecting onto Floor)
# We extend PAST the Floor depth, looking AT the Front Wall
calculate_projection(
    boundary_dist=DIST_TO_FLOOR, 
    perpendicular_dist=DIST_TO_FRONT_WALL, 
    direction_name="DOWN (Elevation)", 
    surface_name="FLOOR (Distance from Front Wall)"
)

# 3. WIDTH / AZIMUTH (Projecting onto Side Walls)
# We extend PAST the Side Wall width, looking AT the Front Wall
# Note: This assumes you are facing Front and projecting onto the Left/Right walls
calculate_projection(
    boundary_dist=DIST_TO_SIDE_WALL, 
    perpendicular_dist=DIST_TO_FRONT_WALL, 
    direction_name="WIDTH (Azimuth)", 
    surface_name="SIDE WALL (Distance from Front Wall)"
)

print("\n\nNOTE: 'Marker Pos' is the distance measured starting from the corner/wall, moving TOWARDS the subject.")