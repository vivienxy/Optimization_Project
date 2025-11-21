import os
import glob
import random

import vedo
from vedo import Volume, show


base_dir = "./FINAL_BRAIN_ATLAS/nii_exports"
brainmask_folder_name = "BRAIN_MASK"

color_palette = [
    "red", "green", "blue", "yellow", "magenta", "cyan",
    "orange", "purple", "pink", "gold", "lime", "salmon",
    "violet", "turquoise", "teal", "indigo",
]

actors = []

brainmask_dir = os.path.join(base_dir, brainmask_folder_name)
if os.path.isdir(brainmask_dir):
    nii_files = glob.glob(os.path.join(brainmask_dir, "*.nii")) + \
                glob.glob(os.path.join(brainmask_dir, "*.nii.gz"))

    print(f"Found {len(nii_files)} brain mask files.")
    for fpath in nii_files:
        print(f"Loading BRAIN_MASK volume: {os.path.basename(fpath)}")
        vol = Volume(fpath)
        surf = vol.isosurface()
        surf.c("grey").alpha(0.05)
        actors.append(surf)
else:
    print(f"[WARN] BRAIN_MASK folder not found at: {brainmask_dir}")

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
        color = random.choice(color_palette)
        surf.c(color).alpha(1.0)
        actors.append(surf)

print("\nRendering 3D scene...")
plt = show(
    actors,
    axes=1,
    viewup="z",        
    title="Brain Structures (vedo)",
)
