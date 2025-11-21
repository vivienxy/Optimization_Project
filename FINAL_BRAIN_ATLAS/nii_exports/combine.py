import os
import glob
import nibabel as nib
import numpy as np

base_dir = "/Users/vivienyu/Desktop/opt_proj/Optimization_Project/FINAL_BRAIN_ATLAS/nii_exports"
output_dir = os.path.join(base_dir, "COMBINED_NII")

os.makedirs(output_dir, exist_ok=True)

def combine_nii_files_in_folder(folder_path, out_name):
    nii_paths = glob.glob(os.path.join(folder_path, "*.nii")) + \
                glob.glob(os.path.join(folder_path, "*.nii.gz"))

    if not nii_paths:
        print(f"[SKIP] No .nii files in: {folder_path}")
        return

    print(f"\n[COMBINE] {os.path.basename(folder_path)} ({len(nii_paths)} files)")

    combined = None
    affine = None

    for i, p in enumerate(nii_paths):
        nii = nib.load(p)
        data = nii.get_fdata()

        if combined is None:
            combined = np.zeros_like(data)
            affine = nii.affine

        combined = np.maximum(combined, data)

        print(f"  ✓ Added {os.path.basename(p)}")

    out_path = os.path.join(output_dir, out_name)

    nib.save(nib.Nifti1Image(combined.astype(np.uint8), affine), out_path)

    print(f"  → Saved combined file: {out_path}")


for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    if folder == "COMBINED_NII":
        continue

    out_name = folder.lower() + ".nii"

    combine_nii_files_in_folder(folder_path, out_name)

print(f"\nAll folders processed.\nCombined files saved in: {output_dir}")
