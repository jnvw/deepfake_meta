import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img.permute(1, 2, 0))
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
