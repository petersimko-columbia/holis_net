import os
import numpy as np
import tifffile
from scipy.ndimage import zoom
from skimage.transform import rescale


def scale_image_stack(image_stack, scaling_factors):
    #scaled_stack = zoom(image_stack, scaling_factors, order=1)
    scaled_stack = rescale(image_stack, scaling_factors, order=1)
    return scaled_stack


# Define input and output directories
input_dir = 'data/testing_data/source_1024_pruned_norm/'
output_dir = 'data/testing_data/source_1024_pruned_norm_scaled/'

# Define resolution
res = [0.795, 0.9, 2]
#ds = [0.9,0.9,0.9]

# Create the output directory if it does not already exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loop over all files in input directory
for filename in os.listdir(input_dir):
    if not filename.endswith('.tif'):
        continue

    # Read in original tiff stack
    original_stack = tifffile.imread(os.path.join(input_dir, filename))

    # Apply scaling
    scaled_stack = scale_image_stack(original_stack, res)

    # Downsample
    #ds_stack = scale_image_stack(scaled_stack, ds)

    # Create output filename based on original filename
    output_filename = f'{filename[:-4]}_scaled.tif'

    # Write normalized stack to output directory
    tifffile.imwrite(os.path.join(output_dir, output_filename), scaled_stack)
