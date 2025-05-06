import torch
import torch.nn as nn
import torchvision.transforms as T
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image

# Define a ResNet-based encoder for 2-class (real vs fake)
class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        # Replace final fc to output 2 logits (no softmax)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        return self.backbone(x)

# Instantiate model and Grad-CAM extractor on the last conv layer
model = ResNetEncoder(pretrained=True)
model.eval()
cam_extractor = GradCAM(model.backbone, target_layer='layer4')

# Preprocess a sample image
img = Image.open("easy_64_1111.jpg").convert("RGB")
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
input_tensor = transform(img)
input_tensor = input_tensor.unsqueeze(0)      # add batch dim
input_tensor.requires_grad_(True)             # enable gradients

# Forward pass (do NOT use torch.no_grad, since we need gradients for CAM)
scores = model(input_tensor)                  # shape [1,2]
pred_idx = scores.argmax(dim=1).item()        # 0 = real, 1 = fake

# Compute Grad-CAM; note passing raw logits and class index
cam_maps = cam_extractor(class_idx=pred_idx, scores=scores)
cam_map = cam_maps[0].squeeze(0).cpu()        # get single heatmap

# Convert CAM to PIL and overlay on original image
heatmap = T.ToPILImage()(cam_map)
result = overlay_mask(img, heatmap, alpha=0.5)

# (Optional) Save or show the result
result.save("cam_overlay.png")
