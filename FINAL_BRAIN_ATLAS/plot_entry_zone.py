import os
import glob
import vedo
from vedo import Volume, show

base_dir = "./final_nii_files"

brainmask_key = "full_brain"
left_entry_key = os.path.join("entry_zone", "LEFT_ENTRY_ZONE")
right_entry_key = os.path.join("entry_zone", "RIGHT_ENTRY_ZONE")
obstacle_key = "obstacles"

left_stn_key = os.path.join("subthalamic_nucleus", "LEFT_STN")
right_stn_key = os.path.join("subthalamic_nucleus", "RIGHT_STN")

actors = []

brainmask_dir = os.path.join(base_dir, brainmask_key)
if os.path.isdir(brainmask_dir):
    nii_files = glob.glob(os.path.join(brainmask_dir, "**", "*.nii"), recursive=True) + \
                glob.glob(os.path.join(brainmask_dir, "**", "*.nii.gz"), recursive=True)

    print(f"Found {len(nii_files)} brain mask files.")
    for fpath in nii_files:
        relname = os.path.relpath(fpath, base_dir)
        print(f"Loading brain mask: {relname}")

        vol = Volume(fpath)
        surf = vol.isosurface()

        surf.c("grey").alpha(0.05)
        actors.append(surf)

else:
    print(f"[WARN] full_brain not found at: {brainmask_dir}")


for root, dirs, files in os.walk(base_dir):
    for file in files:
        if not (file.endswith(".nii") or file.endswith(".nii.gz")):
            continue

        fpath = os.path.join(root, file)
        relpath = os.path.relpath(fpath, base_dir)

        if brainmask_key in relpath:
            continue

        folder_lower = relpath.lower()

        if left_entry_key.lower() in folder_lower:
            color = "lightblue"
            alpha = 0.6

        elif right_entry_key.lower() in folder_lower:
            color = "plum"
            alpha = 0.6

        elif obstacle_key.lower() in folder_lower:
            color = "red"
            alpha = 0.3

        elif left_stn_key.lower() in folder_lower:
            color = "blue"
            alpha = 1.0

        elif right_stn_key.lower() in folder_lower:
            color = "purple"
            alpha = 1.0

        else:
            color = "white"
            alpha = 0.05

        print(f"Loading structure: {relpath}  →  {color}")

        vol = Volume(fpath)
        surf = vol.isosurface()
        surf.c(color).alpha(alpha)
        actors.append(surf)

print("\nRendering 3D scene...")
plt = show(
    actors,
    axes=1,
    viewup="z",
    title="Brain Structures (vedo)",
)
