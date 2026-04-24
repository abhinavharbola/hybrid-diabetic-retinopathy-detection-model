"""
src/inference.py: Preprocessing, prediction, and Grad-CAM overlay.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import DRDModel

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
IMAGE_SIZE  = 384
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

_TRANSFORM = A.Compose([
    A.Normalize(),   # ImageNet mean/std
    ToTensorV2(),
])


# ──────────────────────────────────────────────
# Ben-Graham sharpening
# ──────────────────────────────────────────────
def _ben_graham(img_bgr: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    sigma = image_size // 40          # 384 // 40 = 9
    blur  = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharp = cv2.addWeighted(img_bgr, 4, blur, -4, 128)
    return sharp


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def preprocess(pil_image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """
    Return (tensor [1,3,H,W], rgb_float32 [H,W,3] for Grad-CAM overlay).
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img_bgr = _ben_graham(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_float = img_rgb.astype(np.float32) / 255.0

    augmented = _TRANSFORM(image=img_rgb)
    tensor = augmented["image"].unsqueeze(0)   # [1,3,H,W]
    return tensor, rgb_float


def predict(
    model: DRDModel,
    tensor: torch.Tensor,
    device: torch.device,
) -> tuple[int, np.ndarray]:
    """
    Run forward pass. Returns (pred_class, probs).
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_class = int(np.argmax(probs))
    return pred_class, probs


# ──────────────────────────────────────────────
# Native Grad-CAM Implementation
# ──────────────────────────────────────────────
class NativeGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to extract feature maps and gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, target_class=None):
        self.model.zero_grad()

        # Ensure input requires gradient for the backward pass
        if not x.requires_grad:
            x.requires_grad = True

        logits = self.model(x)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass for the target class
        score = logits[0, target_class]
        score.backward()

        # Global average pool the gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Multiply activations by gradient weights
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Average over channels and apply ReLU
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Normalize to [0, 1]
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap


def compute_gradcam(
    model: DRDModel,
    tensor: torch.Tensor,
    rgb_float: np.ndarray,
    device: torch.device,
    target_class: int | None = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap blended onto rgb_float. Returns uint8 RGB [H,W,3].
    """
    tensor = tensor.to(device)
    
    # Target the exact layer you were using before
    target_layer = model.cnn.blocks[-1][-1].bn2

    # Initialize native Grad-CAM
    cam = NativeGradCAM(model, target_layer)

    # Generate the raw heatmap
    heatmap = cam(tensor, target_class)

    # Resize heatmap to match the original image dimensions
    heatmap_resized = cv2.resize(heatmap, (rgb_float.shape[1], rgb_float.shape[0]))

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # Convert to RGB
    heatmap_color = np.float32(heatmap_color) / 255.0

    # Blend with the original image (maintaining the previous image_weight=0.55)
    image_weight = 0.55
    overlay = (1 - image_weight) * heatmap_color + image_weight * rgb_float
    
    # Clip values to [0, 1] range
    overlay = np.clip(overlay, 0, 1)

    return np.uint8(255 * overlay)