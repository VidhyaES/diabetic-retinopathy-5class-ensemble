import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from models.model import get_model


# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# GradCAM Class
# ==========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        if cam.max() != 0:
            cam = cam / cam.max()

        return cam


# ==========================================================
# Utility Functions
# ==========================================================
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0

    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return img, tensor.to(device)


def overlay_heatmap(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--fold", type=int, default=1, help="Fold number (1-5)")
    parser.add_argument("--class_idx", type=int, default=None, help="Target class index")
    args = parser.parse_args()

    print("Running Grad-CAM...")
    print("Using device:", device)

    # ------------------------------------------------------
    # Load Model
    # ------------------------------------------------------
    model = get_model().to(device)

    checkpoint_path = f"checkpoints_kfold/model_fold_{args.fold}.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Loaded fold {args.fold} model.")

    # ------------------------------------------------------
    # Target Layer (EfficientNet last block)
    # ------------------------------------------------------
    try:
        target_layer = model._blocks[-1]
    except AttributeError:
        raise RuntimeError("Target layer not found. Check model architecture.")

    # ------------------------------------------------------
    # Load Image
    # ------------------------------------------------------
    original_img, input_tensor = load_image(args.image)

    # ------------------------------------------------------
    # Generate CAM
    # ------------------------------------------------------
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, class_idx=args.class_idx)

    overlay = overlay_heatmap(original_img, heatmap)

    # ------------------------------------------------------
    # Save Output
    # ------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/gradcam_fold{args.fold}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print("Grad-CAM saved to:", output_path)
