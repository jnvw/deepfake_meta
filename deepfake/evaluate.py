import torch
from data.dataset import DeepfakeDataset
from models.encoder import ResNetEncoder
from models.metaoptnet_svm import MetaOptNetSVM
from utils.few_shot_sampler import create_episode
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the test dataset
test_dataset = DeepfakeDataset("data", mode='test')

# Initialize encoder and load pretrained weights
encoder = ResNetEncoder()
encoder.load_state_dict(torch.load("models/encoder_ep4.pth"))
encoder.eval()

classifier = MetaOptNetSVM()

accuracies = []
episodes = 5

def overlay_mask(img, mask, alpha=0.5):
    return cv2.addWeighted(img, alpha, mask, 1 - alpha, 0)

for episode in range(episodes):
    # Sample episode
    support_x, support_y, query_x, query_y = create_episode(test_dataset)

    # Extract support features
    support_features = encoder(support_x)
    support_features = support_features.view(support_x.size(0), -1)
    support_y = support_y.view(-1)

    # Extract query features
    query_features = encoder(query_x)
    preds = classifier(support_features, support_y, query_features)
    pred_labels = preds.argmax(dim=1)
    acc = (pred_labels == query_y).float().mean().item()
    accuracies.append(acc)

    # ---- Grad-CAM visualization ----
    query_image = query_x[0].unsqueeze(0).clone().detach()
    query_image.requires_grad_()  # Enable gradients for Grad-CAM

    for param in encoder.parameters():
        param.requires_grad = False  # Freeze encoder weights

    _ = encoder(query_image)  # Forward pass for Grad-CAM

    # Generate Grad-CAM
    cams = encoder.generate_grad_cam(query_image)  # Should return list/tensor of heatmaps

    heatmap = cams[0].cpu().numpy()
    heatmap = cv2.resize(heatmap, (query_image.shape[2], query_image.shape[3]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    query_np = query_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    query_np = np.uint8(255 * query_np)

    overlay = overlay_mask(query_np, heatmap, alpha=0.5)
    plt.imshow(overlay[..., ::-1])  # Convert BGR to RGB
    plt.title(f"Prediction: {pred_labels[0].item()} | Ground truth: {query_y[0].item()}")
    plt.axis('off')
    plt.show()

    print(f"[Episode {episode}] Accuracy: {acc:.4f}")

# Final results
print(f"\nAverage Accuracy over {episodes} episodes: {sum(accuracies)/len(accuracies):.4f}")
print(f"Max Accuracy over {episodes} episodes: {max(accuracies):.4f}")
print(f"Min Accuracy over {episodes} episodes: {min(accuracies):.4f}")
