import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoModel
import torch.nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Calculate CLIP Compositional Similarity

processor_1 = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_1 = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

image_generated = Image.open("path_to_checkpoint/images/A-photo-of-<asset0>-and-<asset1>-step-500.png")
with torch.no_grad():
    inputs_generated = processor_1(images=image_generated, return_tensors="pt").to(device)
    image_features_generated = model_1.get_image_features(**inputs_generated)

image_original = Image.open("path_to_checkpoint/attention/0-step/image.png")
with torch.no_grad():
    inputs_original = processor_1(images=image_original, return_tensors="pt").to(device)
    image_features_original = model_1.get_image_features(**inputs_original)

cos = nn.CosineSimilarity(dim=0)
sim = cos(image_features_generated[0], image_features_original[0]).item()
sim = (sim+1)/2
print("CLIP Compositional Similarity: ", sim)


# 2. Calculate DINO Compositional Similarity

processor_2 = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_2 = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

with torch.no_grad():
    inputs_generated = processor_2(images=image_generated, return_tensors="pt").to(device)
    outputs_generated = model_2.get_image_features(**inputs_generated)
    image_features_generated_sample = outputs_generated.last_hidden_state
    image_features_generated = image_features_generated_sample.mean(dim=1)

with torch.no_grad():
    inputs_original = processor(images=image_original, return_tensors="pt").to(device)
    outputs_original = model(**inputs_original)
    image_features_original_sample = outputs_original.last_hidden_state
    image_features_original = image_features_original_sample.mean(dim=1)

sim = cos(image_features_generated[0], image_features_original[0]).item()
sim = (sim+1)/2
print("DINO Compositional Similarity: ", sim)


