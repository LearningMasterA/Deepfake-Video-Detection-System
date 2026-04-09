import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model_loader import model


def generate_gradcam(image_tensor, raw_image):

    # Select target layer
    target_layer = model.features[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    input_tensor = image_tensor.unsqueeze(0).to(next(model.parameters()).device)

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # ✅ Resize heatmap to match image
    grayscale_cam = cv2.resize(
        grayscale_cam,
        (raw_image.shape[1], raw_image.shape[0])
    )

    # Convert image to RGB and normalize
    rgb_img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Generate overlay
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
