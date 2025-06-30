import torch
from torchvision import transforms
from PIL import Image
from transformer_net import TransformerNet  # Make sure this file is in same folder

def load_model(model_path="models/candy.pth"):
    model = TransformerNet()
    state_dict = torch.load(model_path, map_location="cpu")

    # Remove keys like running_mean/running_var if present
    for key in list(state_dict.keys()):
        if "running_" in key:
            del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()
    return model

def stylize_image(input_image_path: str, output_image_path: str, model_path="models/candy.pth"):
    input_image = Image.open(input_image_path).convert("RGB")
    
    # Keep original size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))  # [C,H,W] → [1,C,H,W]
    ])
    input_tensor = transform(input_image)

    model = load_model(model_path)

    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0)  # [1,C,H,W] → [C,H,W]

    # Clamp to valid pixel range and convert to image
    output_image = transforms.ToPILImage()(output_tensor.clamp(0, 1))
    output_image.save(output_image_path)
    print(f"Saved stylized image to {output_image_path}")

# === Run it ===
if __name__ == "__main__":
    stylize_image("test_images/amber.jpg", "output.jpg")
