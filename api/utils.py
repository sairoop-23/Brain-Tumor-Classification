import cv2
import numpy as np
from io import BytesIO
from PIL import Image

IMG_SIZE = 150

def preprocess_image(file_bytes):
    # Load image as grayscale
    pil_img = Image.open(BytesIO(file_bytes)).convert("L")  # convert to grayscale
    img = np.array(pil_img)

    # Apply same preprocessing as training
    img = cv2.bilateralFilter(img, 2, 50, 50)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img
