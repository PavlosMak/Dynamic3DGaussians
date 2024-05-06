import os
from PIL import Image
import numpy as np

paths = ["/media/pavlos/One Touch/datasets/our_baseline/torus"]
cameras = [10]

if __name__ == "__main__":
    for i, p in enumerate(paths):
        cams = cameras[i]
        for cam_id in range(cams + 1):
            cam_path = f"{p}/ims/{cam_id}"
            ims = os.listdir(cam_path)
            out = f"{p}/white/{cam_id}"
            os.makedirs(out, exist_ok=True)
            for im in ims:
                if "background" in im:
                    continue
                seg_path = f"{p}/seg/{cam_id}/{im}"
                im_path = f"{p}/ims/{cam_id}/{im}"
                image = np.array(Image.open(im_path))
                mask = np.array(Image.open(seg_path))
                idx = (mask == 0)
                image[idx] = [255, 255, 255]
                image = Image.fromarray(image)
                image.save(f"{out}/{im}")
