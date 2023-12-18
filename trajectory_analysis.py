import matplotlib.pyplot as plt
import numpy as np
import torch

import imageio

import matplotlib.animation as animation
from tqdm import tqdm

from helpers import load_scene_data

OUTPUT_PATH = "./output"


class SceneDeformation:
    def __init__(self, scene_data, is_fg):
        self.scene_data = scene_data
        self.is_fg = is_fg
        self.frame_count = len(scene_data)

        self.frames_origins = [frame["means3D"] for frame in scene_data]
        self.frames_rotations = [frame["rotations"] for frame in scene_data]

        # calculate gradients
        self.positional_gradients = [self.frames_origins[i] - self.frames_origins[i - 1] for i in
                                     range(1, self.frame_count)]
        self.rotational_gradients = [self.frames_rotations[i] - self.frames_rotations[i - 1] for i in
                                     range(1, self.frame_count)]


def plot_vector_field(origins: np.array, directions: np.array, fraction=0.03, multiplier=1, output_path=None):
    # first sample the points to plot
    number_of_kernels = len(origins)
    indices = np.random.choice(number_of_kernels, int(fraction * number_of_kernels), replace=False)
    origins = origins[indices]
    directions = multiplier * directions[indices]

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

    # Create a ScalarMappable for color mapping
    sm = plt.cm.ScalarMappable(cmap=plt.cm.magma)
    sm.set_array(colors)
    plt.colorbar(sm, label='Magnitude')

    ax.quiver(x, y, z, u, v, w, length=0.1, color=plt.cm.magma(colors))
    plt.title(f"Deformation Gradients - {len(directions)} points")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    return lines


def animate_gradients(scene: SceneDeformation):
    positional_gradients_tensor = torch.stack(scene.positional_gradients)
    # transpose to group gradients per point instead of per frame
    walks = torch.transpose(positional_gradients_tensor, 0, 1)
    # sample points to reduce them for the visualization
    indices = np.random.choice(walks.shape[0], 2000, replace=False)
    walks = walks[indices].cpu()

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create lines initially without data
    lines = [ax.plot([], [], [])[0] for _ in walks]

    # Setting the axes properties
    ax.set(xlabel='X')
    ax.set(ylabel='Y')
    ax.set(zlabel='Z')

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_lines, scene.frame_count, fargs=(walks, lines), interval=100)

    plt.show()


if __name__ == "__main__":
    model_location = "/media/pavlos/One Touch/datasets/dynamic_3d_output/"
    exp = "test_experiment2"
    seq = "basketball"

    scene_data, is_fg = load_scene_data(seq, exp, False, model_location)

    scene = SceneDeformation(scene_data, is_fg)

    plot_vector_field(scene.frames_origins[0].cpu(), scene.positional_gradients[0].cpu(),
                          fraction=0.025, multiplier=10)

    # animate_gradients(scene)
    # images = []
    # for i in tqdm(range(scene.frame_count - 1)):
    #     output_path = f"{OUTPUT_PATH}/{i}.jpg"
    #     plot_vector_field(scene.frames_origins[i].cpu(), scene.positional_gradients[i].cpu(),
    #                       fraction=0.025, multiplier=10,
    #                       output_path=output_path)
    #     images.append(imageio.imread(output_path))
    # imageio.mimsave(f"{OUTPUT_PATH}/gradients.gif", images)
