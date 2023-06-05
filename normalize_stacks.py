import os
import numpy as np
import tifffile

def normalize_image_stack(image_stack):
    mean = np.mean(image_stack)
    std = np.std(image_stack)
    normalized_stack = (image_stack - mean) / std
    return normalized_stack

# Define input and output directories
input_dir = 'data/testing_data/source_1024_pruned/'
output_dir = 'data/testing_data/source_1024_pruned_norm/'

# Create the output directory if it does not already exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loop over all files in input directory
for filename in os.listdir(input_dir):
    if not filename.endswith('.tif'):
        continue

    # Read in original tiff stack
    original_stack = tifffile.imread(os.path.join(input_dir, filename))

    # Apply normalization
    normalized_stack = normalize_image_stack(original_stack)

    # Create output filename based on original filename
    output_filename = f'{filename[:-4]}_norm.tif'

    # Write normalized stack to output directory
    tifffile.imwrite(os.path.join(output_dir, output_filename), normalized_stack)
