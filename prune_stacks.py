import os
import tifffile

# Define input and output directories
input_dir = 'data/testing_data/source_1024/'
output_dir = 'data/testing_data/source_1024_pruned/'

# Create the output directory if it does not already exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Loop over all files in input directory
for filename in os.listdir(input_dir):
    if not filename.endswith('.tif'):
        continue

    # Read in original tiff stack
    original_stack = tifffile.imread(os.path.join(input_dir, filename))

    # Prune the stack by extracting frames 200-600
    pruned_stack = original_stack[200:401]

    # Create output filename based on original filename
    output_filename = f'{filename[:-4]}_pruned.tif'

    # Write pruned stack to output directory
    tifffile.imwrite(os.path.join(output_dir, output_filename), pruned_stack)
