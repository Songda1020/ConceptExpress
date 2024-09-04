import torch
from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, CLIPModel, AutoModel
import torch.nn
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Calculate CLIP Compositional Similarity

processor_1 = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# image_generated = Image.open("../ckpts/test/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")
# image_generated = Image.open("../ckpts/test_2/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")
# image_generated = Image.open("../ckpts/test_4/images/A-photo-of-_asset0_-and-_asset1_-step-500.png")
# image_generated = Image.open("../ckpts/test_4/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")
image_generated = Image.open("../ckpts/03_1/images/A-photo-of-_asset0_-and-_asset1_-and-_asset2_-and-_asset3_-step-500.png")
# image_generated = Image.open("../ckpts/03_2/images/A-photo-of-<asset0>-and-<asset1>-and-<asset2>-and-<asset3>-step-500.png")
image_original = Image.open("../ckpts/03_1/attention/0-step/image.png")

width, height = image_generated.size

n = 4
sim_list = []
for i in range(n):
    image_generated_isolated = image_generated.crop((width/4*i, 0, width/4*(i+1), height))

    with torch.no_grad():
        inputs_generated = processor_1(images=image_generated_isolated, return_tensors="pt").to(device)
        image_features_generated = model_1.get_image_features(**inputs_generated)

    with torch.no_grad():
        inputs_original = processor_1(images=image_original, return_tensors="pt").to(device)
        image_features_original = model_1.get_image_features(**inputs_original)

    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features_generated[0], image_features_original[0]).item()
    sim = (sim+1)/2
    sim_list.append(sim)
    
print("CLIP Compositional Similarity: ", sum(sim_list) / len(sim_list))

# 2. Calculate DINO Compositional Similarity

processor_2 = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_2 = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

n = 4
sim_list = []
for i in range(n):
    image_generated_isolated = image_generated.crop((width/4*i, 0, width/4*(i+1), height))
    
    with torch.no_grad():
        inputs_generated = processor_2(images=image_generated_isolated, return_tensors="pt").to(device)
        outputs_generated = model_2(**inputs_generated)
        image_features_generated_sample = outputs_generated.last_hidden_state
        image_features_generated = image_features_generated_sample.mean(dim=1)

    with torch.no_grad():
        inputs_original = processor_2(images=image_original, return_tensors="pt").to(device)
        outputs_original = model_2(**inputs_original)
        image_features_original_sample = outputs_original.last_hidden_state
        image_features_original = image_features_original_sample.mean(dim=1)

    sim = cos(image_features_generated[0], image_features_original[0]).item()
    sim = (sim+1)/2
    sim_list.append(sim)

print("DINO Compositional Similarity: ", sum(sim_list) / len(sim_list))



