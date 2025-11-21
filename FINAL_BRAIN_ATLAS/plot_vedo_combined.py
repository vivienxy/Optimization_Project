import os
import glob
import random

import vedo
from vedo import Volume, show

base_dir = "/Users/vivienyu/Desktop/opt_proj/FINAL_BRAIN_ATLAS/nii_exports/COMBINED_NII"

brainmask_file_name = "brain_mask.nii"
color_palette = [
    "red", "green", "blue", "yellow",
    "orange", "purple", "pink"
]

actors = []

# ---------- 1. Load brain as translucent grey ----------
brainmask_path_nii = os.path.join(base_dir, brainmask_file_name)
brainmask_path_niigz = brainmask_path_nii + ".gz"

brainmask_path = None
if os.path.isfile(brainmask_path_nii):
    brainmask_path = brainmask_path_nii
elif os.path.isfile(brainmask_path_niigz):
    brainmask_path = brainmask_path_niigz

if brainmask_path is not None:
    print(f"Loading BRAIN_MASK volume: {os.path.basename(brainmask_path)}")
    vol = Volume(brainmask_path)
    surf = vol.isosurface()
    surf.c("grey").alpha(0.05)
    actors.append(surf)
else:
    print(f"[WARN] Brain mask file not found in: {base_dir}")
    print(f"  Expected: {brainmask_path_nii} or {brainmask_path_niigz}")

# ---------- 2. Load subcortcal structures ----------
nii_files = glob.glob(os.path.join(base_dir, "*.nii")) + \
            glob.glob(os.path.join(base_dir, "*.nii.gz"))

nii_files = [f for f in nii_files if os.path.basename(f) not in {
    os.path.basename(brainmask_path_nii),
    os.path.basename(brainmask_path_niigz),
}]

print(f"\nFound {len(nii_files)} non-brainmask structure file(s) in COMBINED_NII.")

for fpath in nii_files:
    fname = os.path.basename(fpath)
    print(f"  Loading structure: {fname}")
    vol = Volume(fpath)
    surf = vol.isosurface()
    color = random.choice(color_palette)
    surf.c(color).alpha(1.0)
    actors.append(surf)

# ---------- 3. display ----------
print("\nRendering 3D scene...")
plt = show(
    actors,
    axes=1,
    viewup="z",
    title="Brain Structures (Combined NIfTI, vedo)",
)
