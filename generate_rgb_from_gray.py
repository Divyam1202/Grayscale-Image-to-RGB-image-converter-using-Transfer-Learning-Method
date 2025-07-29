import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

# Load pretrained model
model = load_model("model1_best.h5")

# Input/output folders
input_folder = r"C:\Users\Divyam Chandak\Desktop\raw_rgb_gray\grayscale_pngs"
output_folder = r"C:\Users\Divyam Chandak\Desktop\raw_rgb_gray\generated_rgb"
os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(input_folder):
    if not fname.endswith('.png'):
        continue

    # Load and preprocess grayscale
    img_path = os.path.join(input_folder, fname)
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    original_shape = gray_img.shape
    gray_resized = cv2.resize(gray_img, (256, 256))
    input_tensor = gray_resized.reshape(1, 256, 256, 1) / 255.0

    # Predict RGB
    prediction = model.predict(input_tensor)[0]
    rgb_image = (prediction * 255).astype(np.uint8)
    rgb_resized = cv2.resize(rgb_image, (original_shape[1], original_shape[0]))

    # Save result
    out_path = os.path.join(output_folder, fname.replace(".png", "_rgb.png"))
    cv2.imwrite(out_path, cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))

    # Optional: show image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title("Input Grayscale")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_resized)
    plt.title("Predicted RGB")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

print(f"✅ All RGB images saved to: {output_folder}")
