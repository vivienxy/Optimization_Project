import os
import glob
import random

import vedo
from vedo import Volume, show

# ================== EDIT THIS ==================
# Folder that contains BRAIN_MASK, SULCI, VENTRICLES, etc.
base_dir = "/Users/vivienyu/Desktop/opt_proj/FINAL_BRAIN_ATLAS/nii_exports"
# ==============================================

brainmask_folder_name = "BRAIN_MASK"

# Some nice distinct colors for non-brainmask structures
color_palette = [
    "red", "green", "blue", "yellow", "magenta", "cyan",
    "orange", "purple", "pink", "gold", "lime", "salmon",
    "violet", "turquoise", "teal", "indigo",
]

actors = []

# ---------- 1. Load BRAIN_MASK as translucent grey ----------
brainmask_dir = os.path.join(base_dir, brainmask_folder_name)
if os.path.isdir(brainmask_dir):
    nii_files = glob.glob(os.path.join(brainmask_dir, "*.nii")) + \
                glob.glob(os.path.join(brainmask_dir, "*.nii.gz"))

    print(f"Found {len(nii_files)} brain mask files.")
    for fpath in nii_files:
        print(f"Loading BRAIN_MASK volume: {os.path.basename(fpath)}")
        vol = Volume(fpath)
        surf = vol.isosurface()   # labelmaps should work fine with default iso
        surf.c("grey").alpha(0.05)
        actors.append(surf)
else:
    print(f"[WARN] BRAIN_MASK folder not found at: {brainmask_dir}")

# ---------- 2. Load all other folders as opaque random colors ----------
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    if folder_name == brainmask_folder_name:
        continue
    nii_files = glob.glob(os.path.join(folder_path, "*.nii")) + \
                glob.glob(os.path.join(folder_path, "*.nii.gz"))

    if not nii_files:
        continue

    print(f"\nFolder: {folder_name} — {len(nii_files)} file(s)")
    for fpath in nii_files:
        fname = os.path.basename(fpath)
        print(f"  Loading structure: {fname}")
        vol = Volume(fpath)
        surf = vol.isosurface()
        # Random opaque color
        color = random.choice(color_palette)
        surf.c(color).alpha(1.0)
        actors.append(surf)

# ---------- 3. Show everything together ----------
print("\nRendering 3D scene...")
plt = show(
    actors,
    axes=1,            # add axes
    viewup="z",        
    title="Brain Structures (vedo)",
)
