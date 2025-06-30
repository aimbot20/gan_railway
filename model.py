import torch
from torchvision import transforms
from PIL import Image
import os

from transformer_net import TransformerNet

# Load the pre-trained style transfer model
def load_model(model_path="models/candy.pth"):
    model = TransformerNet()
    state_dict = torch.load(model_path, map_location="cpu")

    # Remove deprecated running_* keys if any
    for key in list(state_dict.keys()):
        if "running_" in key:
            del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()
    return model

print("Loading model...")
model = load_model()
print("Model loaded.")

def stylize_image(input_image: Image.Image) -> Image.Image:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))  # [C,H,W] → [1,C,H,W]
    ])

    input_tensor = transform(input_image)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0)  # [1,C,H,W] → [C,H,W]

    output_image = transforms.ToPILImage()(output_tensor.clamp(0, 1))
    return output_image
