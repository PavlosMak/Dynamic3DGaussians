import os
import shutil
import subprocess

import numpy as np
import pymeshlab


def save_obj(file_path, face_indices, vertex_positions):
    with open(file_path, 'w') as file:
        for vertex in vertex_positions:
            file.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))

        for face in face_indices:
            file.write("f")
            for vertex_index in face:
                file.write(" {}".format(vertex_index + 1))  # Adding 1 to convert to 1-based indexing used in .obj files
            file.write("\n")


# GENERATE SOURCE FROM MESH
ms = pymeshlab.MeshSet()
ms.load_new_mesh("/media/pavlos/One Touch/datasets/gt_generation/royal-field/meshing/clean_mesh.obj")
# source_verts = ms.current_mesh().vertex_matrix()
faces = ms.current_mesh().face_matrix()

frames = "/media/pavlos/One Touch/datasets/gt_generation/royal-field/registered_sliding.npz"
frames = np.load(frames)["arr_0"]

path_to_meshes = "/media/pavlos/One Touch/datasets/gt_generation/royal-field/registered_meshes"
for i in range(len(frames)):
    verts = frames[i]
    path_to_mesh = f"{path_to_meshes}/registered_{i}.obj"
    save_obj(path_to_mesh, faces, verts)