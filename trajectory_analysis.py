import matplotlib.pyplot as plt
import numpy as np
from helpers import load_scene_data


def plot_vector_field(origins: np.array, directions: np.array, fraction=0.03):
    # first sample the points to plot
    number_of_kernels = len(origins)
    indices = np.random.choice(number_of_kernels, int(fraction * number_of_kernels), replace=False)
    origins = origins[indices]
    directions = directions[indices]

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = origins[:, 0]
    y = origins[:, 1]
    z = origins[:, 2]

    u = directions[:, 0]
    v = directions[:, 1]
    w = directions[:, 2]

    # color based on the direction magnitude
    colors = np.linalg.norm(directions, axis=1)

    ax.quiver(x, y, z, u, v, w, length=0.1, color=plt.cm.magma(colors))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    model_location = "/media/pavlos/One Touch/datasets/dynamic_3d_output/"
    exp = "test_experiment2"
    seq = "basketball"

    scene_data, is_fg = load_scene_data(seq, exp, False, model_location)

    frame_0 = scene_data[0]
    frame_1 = scene_data[1]

    origins = frame_0["means3D"].cpu()
    gradients = (frame_1["means3D"] - frame_0["means3D"]).cpu()

    plot_vector_field(origins, gradients)
