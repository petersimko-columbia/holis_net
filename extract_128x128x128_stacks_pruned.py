import os
import numpy as np
import tifffile

# define input and output directories
input_dir = 'data/training_data/target/'
output_dir = 'data/training_data/target_128_pruned/'

# Create the output directory if it does not already exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# define chunk size
chunk_size = (128, 128, 128)

# loop over all files in input directory
for filename in os.listdir(input_dir):
    if not filename.endswith('.tif'):
        continue

    # read in original tiff stack and prune
    original_stack = tifffile.imread(os.path.join(input_dir, filename))[200:600]

    # calculate number of chunks in each dimension
    num_chunks = [int(np.ceil(dim / chunk_size[i])) for i, dim in enumerate(original_stack.shape)]

    # loop over all chunks
    for z in range(num_chunks[0]):
        for y in range(num_chunks[1]):
            for x in range(num_chunks[2]):
                # calculate chunk indices
                z_start, z_stop = z * chunk_size[0], (z + 1) * chunk_size[0]
                y_start, y_stop = y * chunk_size[1], (y + 1) * chunk_size[1]
                x_start, x_stop = x * chunk_size[2], (x + 1) * chunk_size[2]

                # slice out chunk from original stack
                chunk = original_stack[z_start:z_stop, y_start:y_stop, x_start:x_stop]

                # check if chunk is smaller than chunk_size
                if chunk.shape != chunk_size:
                    continue
                # create output filename based on original filename and chunk indices
                output_filename = f'{filename[:-4]}_{z_start}_{y_start}_{x_start}.tif'

                # write chunk to output directory
                tifffile.imwrite(os.path.join(output_dir, output_filename), chunk)