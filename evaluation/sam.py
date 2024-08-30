import torch
from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, CLIPModel, AutoModel
import torch.nn
from torch import nn
import numpy as np
# import sys
# sys.path.append("..")

from segment_anything import sam_model_registry, SamPredictor
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image_generated = Image.open("../ckpts/test/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")
# image_generated = Image.open("../ckpts/test_2/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")
# image_generated = Image.open("../ckpts/test_4/images/A-photo-of-_asset0_-and-_asset1_-step-500.png")
# image_generated = Image.open("../ckpts/test_4/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")

# image_original = Image.open("../ckpts/test/attention/0-step/image.png")
# image_original = Image.open("../ckpts/test_2/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")
# image_original = Image.open("../temp_ckpts/test_2/attention/0-step/image.png")
image_original = Image.open("../temp_ckpts/test_4/attention/0-step/image.png")

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# inputs_mask_1 = Image.open("../ckpts/test_4/attention/0-step/final_attention1.png")
# inputs_mask_1 = Image.open("../ckpts/test_4/attention/0-step/final_attention0.png")
# inputs_mask_1 = Image.open("../ckpts/test_4/attention/0-step/final_attention0.png")
# inputs_mask_1 = Image.open("../temp_ckpts/test_2/attention/0-step/final_attention0.png")
# inputs_mask_1 = Image.open("../temp_ckpts/test_2/attention/0-step/final_attention1.png")
inputs_mask_1 = Image.open("../temp_ckpts/test_4/attention/0-step/final_attention0.png")
# inputs_mask_1 = Image.open("../temp_ckpts/test_4/attention/0-step/final_attention1.png")
tsfm = transforms.Resize([256,256])
image_np = np.array(tsfm(image_original))
tsfm_mask = transforms.Resize([256,256], interpolation=transforms.InterpolationMode.NEAREST)
inputs_mask_1_np = np.array(tsfm_mask(inputs_mask_1)).transpose(0, 1).reshape(256,256) / 255
# print(np.sum(inputs_mask_1_np, axis=0))
# print(np.sum(inputs_mask_1_np, axis=1))
indices = np.argwhere(inputs_mask_1_np > 0)
indices = indices[:, ::-1]

row_rand_array = np.arange(indices.shape[0])
np.random.shuffle(row_rand_array)
selected_indices = indices[row_rand_array[:3]]
# selected_indices = indices[:300]

# input_point = np.array([[500, 375], [1125, 625]])
# input_label = np.array([1, 1])
input_point = selected_indices
input_label = np.array([1] * 3)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image_np)

masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    # box=input_box,
    multimask_output=False,
)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def obtain_masked_image(mask, ax):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)
    ax.imshow(mask_image)

plt.figure(figsize=(10,10))
plt.imshow(image_np)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
# show_mask(inputs_mask_1_np, plt.gca())
# plt.axis('on')
plt.savefig("./masked_image_output.png")

output_mask_np = np.squeeze(masks)
output_mask = Image.fromarray(output_mask_np)
output_mask.save("./mask_output.png")

h, w = output_mask_np.shape[-2:]
output_image_masked_np = image_np * output_mask_np.reshape(h, w, 1)
# output_image_masked = Image.fromarray(output_image_masked_np[:, :, ::-1])
output_image_masked = Image.fromarray(output_image_masked_np)
# output_image_masked.save("./output_image_masked.png")
# output_image_masked.save("./sam_original_masked_1.png")
# output_image_masked.save("./sam_original_masked_0.png")
# output_image_masked.save("./sam_generated_test_2_masked_0.png")
# output_image_masked.save("./sam_generated_test_2_masked_1.png")
# output_image_masked.save("./sam_generated_test_4_masked_0.png")
output_image_masked.save("./sam_generated_test_4_masked_0.png")

sim_i = None



