import os
import tifffile
import numpy as np
import cv2

def tiff_stacker(dir_path,output_path):
    # Get a list of all TIFF files in the directory
    tiff_files = [file for file in os.listdir(dir_path) if file.endswith('.tif')]

    # Read the first image to determine its shape
    first_image = tifffile.imread(os.path.join(dir_path, tiff_files[0]))

    # Initialize an empty stack with the appropriate dimensions
    stack = np.zeros((len(tiff_files), first_image.shape[0], first_image.shape[1]), dtype=np.uint16)

    # Read each TIFF file and add it to the stack
    for i, tiff_file in enumerate(tiff_files):
        image = tifffile.imread(os.path.join(dir_path, tiff_file))
        image[image < 3200] = 0
        image = image.clip(min=0)  # Ensure pixel values are not negative

        # Display the output frame by frame
        cv2.imshow('output', image)
        cv2.waitKey(1)  # Add this line to update the displayed image

        stack[i, :, :] = image

    # Save the stack as a multi-image TIFF file
    tifffile.imsave(output_path, stack)

    print(f"Multi-image TIFF stack saved at {output_path}")

input_dir = r""
output_dir = r""
for foldername in os.listdir(input_dir):
    input_path = os.path.join(input_dir, foldername)
    output_path = os.path.join(output_dir, foldername + 'stack.tiff')
    tiff_stacker(input_path, output_path)