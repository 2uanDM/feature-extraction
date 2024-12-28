import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from src.model import ReIDSiamese
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ReIDSiamese().to(device)
model.load_state_dict(
    torch.load(
        "/home/quan/workspace/bkai/feature-extraction/person_reid/model/20241228_021446_siamese_model.pth"
    )
)

transform = transforms.Compose(
    [
        transforms.Resize((160, 80)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def process_images(input1_path, input2_path, model, device):
    try:
        # Validate inputs
        if not input1_path or not input2_path:
            raise ValueError("Input paths cannot be empty")

        # Read images with error handling
        try:
            input1 = Image.open(input1_path).convert("RGB")
            input2 = Image.open(input2_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to load images: {str(e)}")

        # Ensure model is in eval mode
        model.eval()

        # Transform and move to device
        with torch.no_grad():
            input1_tensor = transform(input1).unsqueeze(0).to(device)
            input2_tensor = transform(input2).unsqueeze(0).to(device)

            # Forward pass
            start_time = time.perf_counter()
            output = model(input1_tensor, input2_tensor)
            print(
                f"Time taken to forward: {time.perf_counter() - start_time:.2f} seconds"
            )
            print(f"==>> output: {output}")

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(input1_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
        ax1.set_title("Input 1")
        ax2.imshow(input2_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
        ax2.set_title("Input 2")

        # Save the figure
        plt.savefig("output.png")

        return output

    except Exception as e:
        print(f"Error processing images: {str(e)}")
        raise
    finally:
        plt.close("all")  # Cleanup matplotlib resources
        torch.cuda.empty_cache()  # Clear GPU memory if used


start_time = time.perf_counter()
process_images(
    "/home/quan/workspace/bkai/feature-extraction/2.png",
    "/home/quan/workspace/bkai/feature-extraction/4.png",
    model,
    device,
)
print(f"Time taken: {time.perf_counter() - start_time:.2f} seconds")
