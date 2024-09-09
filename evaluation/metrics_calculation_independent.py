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
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# concept_generated_0 = Image.open("ckpts/3_3/images/A-photo-of-<asset0>-step-500.png")
# concept_generated_1 = Image.open("ckpts/3_3/images/A-photo-of-<asset1>-step-500.png")
# concept_generated_2 = Image.open("ckpts/3_3/images/A-photo-of-<asset2>-step-500.png")
# concept_generated_3 = Image.open("ckpts/3_3/images/A-photo-of-<asset3>-step-500.png")
# concept_generated_4 = Image.open("ckpts/3_3/images/A-photo-of-<asset4>-step-500.png")
# concept_generated_5 = Image.open("ckpts/3_3/images/A-photo-of-<asset5>-step-500.png")

concept_generated_0 = Image.open("ckpts/Spectral_3_1/images/A-photo-of-<asset0>-step-500.png")
concept_generated_1 = Image.open("ckpts/Spectral_3_1/images/A-photo-of-<asset1>-step-500.png")
concept_generated_2 = Image.open("ckpts/Spectral_3_1/images/A-photo-of-<asset2>-step-500.png")
concept_generated_3 = Image.open("ckpts/Spectral_3_1/images/A-photo-of-<asset3>-step-500.png")

concept_generated = concept_generated_3

image_input = Image.open("ckpts/Spectral_3_1/attention/0-step/image.png")
sam_output_dir = "ckpts/Spectral_3_1/attention/"
sam_output_file = "sam_generated_masked_3.png"

# inputs_mask_0 = Image.open("ckpts/3_3/attention/0-step/final_attention0.png")
# inputs_mask_1 = Image.open("ckpts/3_3/attention/0-step/final_attention1.png")
# inputs_mask_2 = Image.open("ckpts/3_3/attention/0-step/final_attention2.png")
# inputs_mask_3 = Image.open("ckpts/3_3/attention/0-step/final_attention3.png")
# inputs_mask_4 = Image.open("ckpts/3_3/attention/0-step/final_attention4.png")
# inputs_mask_5 = Image.open("ckpts/3_3/attention/0-step/final_attention5.png")

inputs_mask_0 = Image.open("ckpts/Spectral_3_1/attention/0-step/final_attention0.png")
inputs_mask_1 = Image.open("ckpts/Spectral_3_1/attention/0-step/final_attention1.png")
inputs_mask_2 = Image.open("ckpts/Spectral_3_1/attention/0-step/final_attention2.png")
inputs_mask_3 = Image.open("ckpts/Spectral_3_1/attention/0-step/final_attention3.png")

inputs_mask = inputs_mask_3
print("asset 3")

tsfm = transforms.Resize([256,256])
image_np = np.array(tsfm(image_input))
tsfm_mask = transforms.Resize([256,256], interpolation=transforms.InterpolationMode.NEAREST)
inputs_mask_np = np.array(tsfm_mask(inputs_mask)).transpose(0, 1).reshape(256,256) / 255

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

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

IoU_max = 0

for i in range(10):
    
    indices = np.argwhere(inputs_mask_np > 0)
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
    
    plt.figure(figsize=(10,10))
    plt.imshow(image_np)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    # show_mask(inputs_mask_1_np, plt.gca())
    
    output_mask_np = np.squeeze(masks, axis=0)
    output_mask = Image.fromarray(output_mask_np)
    # output_mask_np.dtype = int
    intersection = sum(sum((inputs_mask_np + output_mask_np) > 1))
    union = sum(sum((inputs_mask_np + output_mask_np) > 0))
    IoU = intersection / union
    if IoU > IoU_max:
        # plt.axis('on')
        plt.savefig("./masked_image_output.png")
        output_mask.save("./mask_output.png")
        h, w = output_mask_np.shape[-2:]
        output_image_masked_np = image_np * output_mask_np.reshape(h, w, 1)
        output_image_masked = Image.fromarray(output_image_masked_np)
        output_image_masked.save(sam_output_dir + sam_output_file)

    if IoU > 0.35:
        IoU_max = max(IoU_max, IoU)
        break

    IoU_max = max(IoU_max, IoU)

    
#    print(IoU)
print("(", "maximum IoU: ", IoU_max, ")")


# 1. Calculate CLIP Indipendent Similarity

processor_1 = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

sim_list = []
width, height = concept_generated.size
n = 4
for i in range(n):
    concept_generated_isolated = concept_generated.crop((width/4*i, 0, width/4*(i+1), height))
    
    with torch.no_grad():
        inputs_generated = processor_1(images=concept_generated_isolated, return_tensors="pt").to(device)
        image_features_generated = model_1.get_image_features(**inputs_generated)

    concept_original = Image.open(sam_output_dir + sam_output_file)
    with torch.no_grad():
        inputs_original = processor_1(images=concept_original, return_tensors="pt").to(device)
        image_features_original = model_1.get_image_features(**inputs_original)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features_generated[0], image_features_original[0]).item()
    sim = (sim+1)/2
    sim_list.append(sim)

print("CLIP Independent Similarity: ", sum(sim_list) / len(sim_list))


# 2. Calculate DINO Indipendent Similarity

processor_2 = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_2 = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

sim_list = []
width, height = concept_generated.size
n = 4
for i in range(n):
    concept_generated_isolated = concept_generated.crop((width/4*i, 0, width/4*(i+1), height))
    
    with torch.no_grad():
        inputs_generated = processor_2(images=concept_generated_isolated, return_tensors="pt").to(device)
        outputs_generated = model_2(**inputs_generated)
        image_features_generated_sample = outputs_generated.last_hidden_state
        image_features_generated = image_features_generated_sample.mean(dim=1)

    with torch.no_grad():
        inputs_original = processor_2(images=concept_original, return_tensors="pt").to(device)
        outputs_original = model_2(**inputs_original)
        image_features_original_sample = outputs_original.last_hidden_state
        image_features_original = image_features_original_sample.mean(dim=1)

    sim = cos(image_features_generated[0], image_features_original[0]).item()
    sim = (sim+1)/2
    sim_list.append(sim)

print("DINO Independent Similarity: ", sum(sim_list) / len(sim_list))

