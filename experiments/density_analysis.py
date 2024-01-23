from helpers import load_scene_data
import plotly.graph_objects as go

import numpy as np

if __name__ == "__main__":
    model_location = "/media/pavlos/One Touch/datasets/dynamic_3d_output/"
    exp = "logical-pond-15"
    seq = "torus"

    scene_data, is_fg = load_scene_data(seq, exp, False, model_location)

    meshgrid = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
    X, Y, Z = meshgrid

    values = np.cos(X * Y * Z) / (X * Y * Z)

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=0.1,
        isomax=0.8,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=17,  # needs to be a large number for good volume rendering
    ))
    fig.show()
