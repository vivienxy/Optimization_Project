import os
import glob
import vedo
from vedo import Volume, show

base_dir = "/Users/vivienyu/Desktop/opt_proj/FINAL_BRAIN_ATLAS/nii_exports/COMBINED_NII"

brainmask_file_name = "brain_mask.nii"

actors = []

brainmask_path_nii = os.path.join(base_dir, brainmask_file_name)
brainmask_path_niigz = brainmask_path_nii + ".gz"


brainmask_path = None
if os.path.isfile(brainmask_path_nii):
    brainmask_path = brainmask_path_nii
elif os.path.isfile(brainmask_path_niigz):
    brainmask_path = brainmask_path_niigz

if brainmask_path is not None:
    print(f"Loading BRAIN_MASK: {os.path.basename(brainmask_path)}")
    vol = Volume(brainmask_path)
    surf = vol.isosurface()
    surf.c("grey").alpha(0.05)
    actors.append(surf)
else:
    print(f"[WARN] Brain mask not found: {brainmask_path_nii} or .nii.gz")


nii_files = glob.glob(os.path.join(base_dir, "*.nii")) + \
            glob.glob(os.path.join(base_dir, "*.nii.gz"))

nii_files = [
    f for f in nii_files
    if os.path.basename(f) not in {
        os.path.basename(brainmask_path_nii),
        os.path.basename(brainmask_path_niigz)
    }
]

print(f"\nFound {len(nii_files)} other structure(s).")

for fpath in nii_files:
    fname = os.path.basename(fpath)
    fname_lower = fname.lower()

    print(f"  Loading: {fname}")
    vol = Volume(fpath)
    surf = vol.isosurface()

    # -----  colo rules -----
    if "stn" in fname_lower or "subthalamic" in fname_lower:
        color = "green"

    elif ("vent" in fname_lower or
          "ventricle" in fname_lower or
          "corpus" in fname_lower or
          "callosum" in fname_lower or
          "sulc" in fname_lower):
        color = "red"

    else:
        color = "white"

    surf.c(color).alpha(1.0)
    actors.append(surf)

print("\nRendering 3D scene...")
plt = show(
    actors,
    axes=1,
    viewup="z",
    title="Brain Structures (STN=green, ventricles/CC/sulci=red)",
)
