import numpy as np
from plyfile import PlyData

# # Torus
# output_directory = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/torus/ground_truth"
# input_directory = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/torus/ground_truth"

# Elastic 0
output_directory = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_0/ground_truth"
input_directory = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/elastic_0/ground_truth"

num_frames = 16

input_directories = [f"{input_directory}/{i}.ply" for i in range(num_frames)]

def load_ply_file(file_path):
    ply_data = PlyData.read(file_path)
    vertices = ply_data['vertex'].data
    return vertices


if __name__ == "__main__":
    pacnerf_points = []
    for path in input_directories:
        verts = load_ply_file(path)
        verts = np.array([np.array(list(v)) for v in verts])
        pacnerf_points.append(verts)
    pacnerf_points = np.array(pacnerf_points)
    np.savez(f"{output_directory}/ground_truth.npz", pacnerf_points)