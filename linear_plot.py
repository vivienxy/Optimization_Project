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
            surf.c("grey").alpha(0.01)
            brain_mask_structures.append(surf)
    # Obstacles
    elif folder_name == corpus_callosum_folder_name or folder_name == sulci_folder_name or folder_name == ventricles_folder_name:
        for fpath in nii_files:
            fname = os.path.basename(fpath)
            print(f"  Loading structure: {fname}")
            vol = Volume(fpath)
            surf = vol.isosurface()
            surf.c("red").alpha(0.66)
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

# Merge brain mask structures to define bounding box
merged_brain_mask = None
if brain_mask_structures:
    print("Merging brain mask structures...")
    merged_brain_mask = merge(brain_mask_structures)
    merged_brain_mask.c("grey").alpha(0.01)

# Merge obstacle structures for distance calculations
merged_obstacles = None
if obstacle_structures:
    print("Merging obstacle structures...")
    merged_obstacles = merge(obstacle_structures)
    merged_obstacles.c("red").alpha(0.75)

# Merge STN structures to sample target point
merged_stn = None
if stn_structures:
    print("Merging STN structures...")
    merged_stn = merge(stn_structures)
    merged_stn.c("green").alpha(1.0)

## Create linear path plot

# Keep regenerating start points until we collect min_attempts successful paths
max_attempts = 1000
min_attempts = 10
successful_attempts = []  # Store (start_point, end_point, distance) tuples
failed_lines = []

for attempt in range(max_attempts):
    if attempt > min_attempts:
        break
    
    # Select random start (brain surface)
    random_point = merged_brain_mask.generate_random_points(1)
    start_point = tuple(random_point.points[0])
    print(f"Selected start point within brain mask: {start_point}")
    
    # Select random end (STN surface)
    random_point = merged_stn.generate_random_points(1)
    end_point = tuple(random_point.points[0])
    print(f"Selected end point within STN: {end_point}")
    
    # Check for intersection if we have obstacles
    if merged_obstacles:
        intersections = merged_obstacles.intersect_with_line(start_point, end_point)
        
        # If no intersections (empty array or None), line is valid
        if intersections is None or len(intersections) == 0:
            # Compute distance to obstacles
            temp_line = Line([start_point, end_point])
            d = temp_line.distance_to(merged_obstacles)
            if isinstance(d, (list, tuple, np.ndarray)):
                dval = float(np.min(d))
            else:
                dval = float(d)
            
            successful_attempts.append((start_point, end_point, dval))
            print(f"Attempt {attempt+1}: SUCCESS - Distance: {dval:.3f}")
        else:
            # Store failed attempt as brown line
            failed_line = Line([start_point, end_point]).c("purple").alpha(0.3).lw(5)
            failed_lines.append(failed_line)
            print(f"Attempt {attempt+1}: FAIL - Line intersects obstacles")
    else:
        # No obstacles, accept any line
        temp_line = Line([start_point, end_point])
        successful_attempts.append((start_point, end_point, 0.0))
        print(f"Attempt {attempt+1}: SUCCESS (no obstacles)")
else:
    print(f"Warning: Only found {len(successful_attempts)} non-intersecting lines after {max_attempts} attempts")

# Find the best path (maximum distance from obstacles)
best_path = None
suboptimal_lines = []

if successful_attempts:
    # Sort by distance (descending)
    successful_attempts.sort(key=lambda x: x[2], reverse=True)
    
    # Best path
    best_start, best_end, best_dist = successful_attempts[0]
    best_path = Line([best_start, best_end]).c("green").lw(8)
    print(f"\nBest path: distance {best_dist:.3f}")
    
    # All other successful paths as brown
    for start_pt, end_pt, dist in successful_attempts[1:]:
        suboptimal_line = Line([start_pt, end_pt]).c("yellow").alpha(0.3).lw(5)
        suboptimal_lines.append(suboptimal_line)
        print(f"Suboptimal path: distance {dist:.3f}")
else:
    print("No successful paths found!")

# Prepare actors for visualization
actors = []
actors.extend(brain_mask_structures)
if merged_obstacles:
    actors.append(merged_obstacles)
if merged_stn:
    actors.append(merged_stn)
# Add all failed attempts (intersecting paths)
actors.extend(failed_lines)
# Add suboptimal successful paths
actors.extend(suboptimal_lines)
# Add best path on top
if best_path:
    actors.append(best_path)

print(f"Rendering scene with {len(failed_lines)} failed attempts, {len(suboptimal_lines)} suboptimal paths, and 1 optimal path...")
show(actors, axes=1, viewup="z", title="Optimal Path To STN with Obstacle Distance")