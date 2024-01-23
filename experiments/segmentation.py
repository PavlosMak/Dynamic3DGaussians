import numpy as np
from PIL import Image
import rembg
import cv2

path_to_input = "/media/pavlos/One Touch/datasets/dynamic_plush/baby_alien/ims/0/000003.jpg"

print("Loading input")
input = Image.open(path_to_input)

# Convert the input image to a numpy array
input_array = np.array(input)

# Apply background removal using rembg
output_array = np.array(rembg.remove(input_array))

# if pixel is below transparency threshold we set it to black
transparent_pixels = (output_array[:, :, 3] < 0.9 * 255)
output_array[transparent_pixels] = np.array([0, 0, 0, 0])
# get the mask
foreground_pixels = output_array[:, :] != [0, 0, 0, 0]
mask = np.clip(np.sum(output_array, axis=2), 0, 1)
cv2.imwrite("seg.png", mask * 255)

# Create a PIL Image from the output array
output_image = Image.fromarray(output_array)

# Save the output image
output_image.save('output_image.png')
