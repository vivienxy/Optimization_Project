import os
import glob
import random
import numpy as np

import vedo
from vedo import Volume, show, Line, merge


## Load data
base_dir = "./FINAL_BRAIN_ATLAS/nii_exports"
brainmask_folder_name = "BRAIN_MASK"
corpus_callosum_folder_name = "CORPUS_CALLLOSUM"
stn_folder_name = "SUBTHALAMIC_NUCLEUS"
sulci_folder_name = "SULCI"
ventricles_folder_name = "VENTRICLES"

stn_dir = os.path.join(base_dir, stn_folder_name)

stn_structures = []
obstacle_structures = []
brain_mask_structures = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    nii_files = glob.glob(os.path.join(folder_path, "*.nii")) + \
                glob.glob(os.path.join(folder_path, "*.nii.gz"))


    if not nii_files:
        continue

    # Brain mask
    if folder_name == brainmask_folder_name:
        for fpath in nii_files:
            fname = os.path.basename(fpath)
            print(f"  Loading structure: {fname}")
            vol = Volume(fpath)
            surf = vol.isosurface()
            surf.c("grey").alpha(0.05)
            brain_mask_structures.append(surf)
    # Obstacles
    elif folder_name == corpus_callosum_folder_name or folder_name == sulci_folder_name or folder_name == ventricles_folder_name:
        for fpath in nii_files:
            fname = os.path.basename(fpath)
            print(f"  Loading structure: {fname}")
            vol = Volume(fpath)
            surf = vol.isosurface()
            surf.c("red").alpha(0.75)
            obstacle_structures.append(surf)
    # STN
    elif folder_name == stn_folder_name:
        for fpath in nii_files:
            fname = os.path.basename(fpath)
            print(f"  Loading structure: {fname}")
            vol = Volume(fpath)
            surf = vol.isosurface()
            surf.c("green").alpha(1.0)
            stn_structures.append(surf)                


# Merge obstacle structures for distance calculations
merged_obstacles = None
if obstacle_structures:
    print("Merging obstacle structures...")
    merged_obstacles = merge(obstacle_structures)
    merged_obstacles.c("red").alpha(0.75)

## Create linear path plot
# Create a simple straight line
# Use bounds from brain mask or obstacles to define endpoints
if brain_mask_structures:
    brain_mask_mesh = merge(brain_mask_structures)
    xmin, xmax, ymin, ymax, zmin, zmax = brain_mask_mesh.bounds()
elif merged_obstacles:
    xmin, xmax, ymin, ymax, zmin, zmax = merged_obstacles.bounds()
else:
    # Default bounds if nothing loaded
    xmin, xmax, ymin, ymax, zmin, zmax = -100, 100, -100, 100, -100, 100

# Create a straight line from random start to random end point
# Keep regenerating until we get a line that doesn't intersect obstacles
max_attempts = 1000
path_line = None
dval = 0

for attempt in range(max_attempts):
    start_point = (random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(zmin, zmax))
    end_point = (random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(zmin, zmax))
    
    # Check for intersection if we have obstacles
    if merged_obstacles:
        intersections = merged_obstacles.intersect_with_line(start_point, end_point)
        
        # If no intersections (empty array or None), line is valid
        if intersections is None or len(intersections) == 0:
            path_line = Line([start_point, end_point]).c("yellow").lw(6)
            
            # Also compute distance for reporting
            d = path_line.distance_to(merged_obstacles)
            if isinstance(d, (list, tuple, np.ndarray)):
                dval = float(np.min(d))
            else:
                dval = float(d)
            
            print(f"Generated straight line from {start_point} to {end_point}")
            print(f"No intersections with obstacles. Minimum distance: {dval:.3f}")
            break
        else:
            print(f"Attempt {attempt+1}: Line intersects obstacles, regenerating...")
    else:
        # No obstacles, accept any line
        path_line = Line([start_point, end_point]).c("yellow").lw(6)
        print(f"Generated straight line from {start_point} to {end_point}")
        break
else:
    print(f"Warning: Could not find a non-intersecting line after {max_attempts} attempts")

# Prepare actors for visualization
actors = []
actors.extend(brain_mask_structures)
if merged_obstacles:
    actors.append(merged_obstacles)
actors.extend(stn_structures)
if path_line:
    actors.append(path_line)

print("Rendering scene...")
show(actors, axes=1, viewup="z", title="Straight Line Path Through Brain with Obstacle Distance")