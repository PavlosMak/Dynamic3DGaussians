import os
import shutil
import subprocess

import numpy as np
import pymeshlab


def call_bcpd(output_dir, target, source, output_name="output_y.txt"):
    starting_path = os.getcwd()
    os.chdir(output_dir)
    subprocess.call([path_to_bcpd, "-x", target, "-y", source])
    os.rename(f"{output_dir}/output_y.txt", f"{output_dir}/{output_name}")
    os.chdir(starting_path)


base_path = "/media/pavlos/One Touch/datasets/gt_generation/royal-field"

path_to_bcpd = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/bcpd/bcpd"

path_to_targets = f"{base_path}/trajectories"

targets = [f"{path_to_targets}/target_{i}.txt" for i in range(0, 7)]

# GENERATE SOURCE FROM MESH
# ms = pymeshlab.MeshSet()
# ms.load_new_mesh("/media/pavlos/One Touch/datasets/gt_generation/royal-field/meshing/clean_mesh.obj")
# source_verts = ms.current_mesh().vertex_matrix()
# np.savetxt(source, source_verts, delimiter=",")

source = f"{base_path}/source.txt"

slide_mesh = True

for i, target in enumerate(targets):
    if slide_mesh and i > 0:
        source = f"{base_path}/registered/registered_{i - 1}.txt"
    call_bcpd(f"{base_path}/registered", target, source,
              output_name=f"registered_{i}.txt")

registered_results = [f"{base_path}/registered/registered_{i}.txt" for i
                      in range(0, 7)]

registered_positions = []
for registered in registered_results:
    registered_positions.append(np.loadtxt(registered, delimiter="\t"))
registered_positions = np.array(registered_positions)
np.savez(f"{base_path}/registered_sliding.npz", registered_positions)
