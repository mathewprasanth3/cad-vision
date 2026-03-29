"""
src/utils/gradcam.py

Grad-CAM implementation for CADVision.
Generates heatmap overlays showing which regions of each view
triggered the model's prediction.

Usage:
    from src.utils.gradcam import GradCAM
    gcam = GradCAM(model)
    heatmaps = gcam.generate(views_tensor, predicted_class_idx)
"""

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import cv2


class GradCAM:

    def __init__(self, model):
        self.model = model
        self.model.eval()

        # storage for forward and backward hooks
        self.feature_maps = None
        self.gradients = None

        # attach hooks to the last conv layer of ResNet (layer4)
        target_layer = model.branch.features[-2]  # ResNet layer4
        self._register_hooks(target_layer)

    def _register_hooks(self, layer):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        views: torch.Tensor,
        class_idx: int,
        pil_images: list,
    ) -> list:
        """
        Generates Grad-CAM heatmap overlays for all 4 views.

        Args:
            views:      [1, 4, 3, 224, 224] tensor
            class_idx:  predicted class index
            pil_images: list of 4 original PIL Images

        Returns:
            list of 4 PIL Images with heatmap overlaid
        """
        batch_size, num_views, c, h, w = views.shape

        # reshape all views for one forward pass
        x = views.view(batch_size * num_views, c, h, w)
        x.requires_grad_(True)

        # forward pass — must be outside no_grad for gradients
        self.model.zero_grad()
        features = self.model.branch(x)                      # [4, 512]
        features_reshaped = features.view(batch_size, num_views, 512)
        pooled, _ = torch.max(features_reshaped, dim=1)      # [1, 512]
        output = self.model.classifier(pooled)               # [1, 24]

        # backward pass for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # feature maps: [4, 512, 7, 7]
        # gradients:    [4, 512, 7, 7]
        feature_maps = self.feature_maps  # [4, 512, 7, 7]
        gradients = self.gradients        # [4, 512, 7, 7]

        # global average pool gradients over spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # [4, 512, 1, 1]

        # weighted combination of feature maps
        cam = (weights * feature_maps).sum(dim=1)             # [4, 7, 7]
        cam = F.relu(cam)                                     # keep positive

        heatmap_images = []
        for i in range(num_views):
            cam_view = cam[i].cpu().numpy()                   # [7, 7]

            # normalize to 0-255
            cam_view = cam_view - cam_view.min()
            if cam_view.max() > 0:
                cam_view = cam_view / cam_view.max()
            cam_view = (cam_view * 255).astype(np.uint8)

            # resize to 224x224
            cam_resized = cv2.resize(cam_view, (224, 224))

            # apply colormap
            heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # overlay on original image
            original = np.array(pil_images[i].resize((224, 224)))
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

            heatmap_images.append(Image.fromarray(overlay))

        return heatmap_images


if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from src.utils.infer import CADVisionPredictor, render_stl, TRANSFORM
    from src.utils.gradcam import GradCAM

    stl_path = "data/raw/stl/0_Oring/0_1.STL"

    predictor = CADVisionPredictor()
    pil_images = render_stl(stl_path)

    tensors = [TRANSFORM(img) for img in pil_images]
    views = torch.stack(tensors).unsqueeze(0).to(predictor.device)

    result = predictor.predict(stl_path)
    print(f"predicted: {result['predicted_class']} ({result['confidence']*100:.1f}%)")

    gcam = GradCAM(predictor.model)
    heatmaps = gcam.generate(views, list(range(24))[0], pil_images)

    for i, (name, img) in enumerate(zip(["front", "side", "top", "isometric"], heatmaps)):
        img.save(f"gradcam_{name}.png")
        print(f"saved gradcam_{name}.png")