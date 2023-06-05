import torch
import tifffile
import os
import numpy as np
from torchvision.transforms import functional as F
import main as m
from unet_class import UNet3D


# Create the output directory if it does not already exist

predictions_dir = 'predictions/'
if not os.path.exists(predictions_dir):
    os.mkdir(predictions_dir)
# Load the TIFF stack
stack = tifffile.imread('data/testing_data/humanBrain1_dualCamera_z01_y05_ch1_cropped_1024.tif').astype('float32')
stack_tensor = torch.from_numpy(stack)
stack_tensor = stack_tensor.unsqueeze(0)
# Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

# Load the trained model weights and create the model architecture
model = UNet3D().to(device)
model.load_state_dict(torch.load(m.model_path))
model.eval()

# Pass the tensor through the model to obtain predictions
with torch.no_grad():
    prediction = model(stack_tensor)

threshold=0.5
# Convert the predictions to a numpy array
prediction = prediction.cpu().numpy()
prediction = np.interp(prediction, (prediction.min(), prediction.max()), (0, 255))
binary_mask = (prediction > threshold).astype('float32')

# Postprocess the predictions to obtain the final segmentation mask
#segmentation_mask = postprocess(numpy_predictions)
tifffile.imwrite(os.path.join(predictions_dir, '/mask_humanBrain1_dualCamera_z01_y05_ch1_cropped_1024.tif'), binary_mask, compress=9)