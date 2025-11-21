from vedo import Volume, VolumeToMesh

vol = Volume("brain_mask.nii")
mesh = VolumeToMesh(vol)
mesh.write("brain_surface.stl")
