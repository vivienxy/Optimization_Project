import os
import shutil

# Current working directory
cwd = os.getcwd()

# Destination folders
destinations = {
    "ventricle": "VENTRICLES",
    "subthalamic_nucleus": "SUBTHALAMIC_NUCLEUS",
    "corpus_callosum": "CORPUS_CALLOSUM",
}

# Special folders
sulci_folder = "SULCI"
brainmask_folder = "BRAIN_MASK"

# Create all destination folders
os.makedirs(os.path.join(cwd, sulci_folder), exist_ok=True)
os.makedirs(os.path.join(cwd, brainmask_folder), exist_ok=True)
for folder in destinations.values():
    os.makedirs(os.path.join(cwd, folder), exist_ok=True)

# Loop through files
for filename in os.listdir(cwd):
    fullpath = os.path.join(cwd, filename)

    # Skip directories
    if os.path.isdir(fullpath):
        continue

    fname_lower = filename.lower()

    # ---------- 1. SPECIAL RULE FOR SULCI ----------
    if "sulcus" in fname_lower and "gyrus" not in fname_lower:
        print(f"Moving: {filename} → {sulci_folder}/")
        shutil.move(fullpath, os.path.join(cwd, sulci_folder, filename))
        continue

    # ---------- 2. RULES FOR SPECIFIC STRUCTURES ----------
    moved = False
    for keyword, folder in destinations.items():
        if keyword in fname_lower:
            print(f"Moving: {filename} → {folder}/")
            shutil.move(fullpath, os.path.join(cwd, folder, filename))
            moved = True
            break

    # ---------- 3. EVERYTHING ELSE → BRAIN_MASK ----------
    if not moved:
        print(f"Moving: {filename} → {brainmask_folder}/")
        shutil.move(fullpath, os.path.join(cwd, brainmask_folder, filename))

print("Done! All files sorted successfully.")
