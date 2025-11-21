from vedo import Volume, Mesh

vol = Volume("brain_mask.nii")

mesh = vol.isosurface(0.5)
mesh.compute_normals().smooth()

mesh.write("brain_surface.stl")
print("Saved brain_surface.stl")
