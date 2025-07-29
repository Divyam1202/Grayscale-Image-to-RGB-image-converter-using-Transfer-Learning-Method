import os
import rawpy
import numpy as np
import imageio

# Set input/output folders
dng_folder = r"C:\Users\Divyam Chandak\Desktop\raw_rgb_gray"
output_folder = os.path.join(dng_folder, "grayscale_pngs")
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(dng_folder):
    if filename.lower().endswith(".dng"):
        path = os.path.join(dng_folder, filename)
        with rawpy.imread(path) as raw:
            raw_image = raw.raw_image.astype(np.float32)
            normalized = 255 * (raw_image - raw_image.min()) / (raw_image.max() - raw_image.min())
            gray_image = normalized.astype(np.uint8)

            # Save as PNG
            output_path = os.path.join(output_folder, filename.replace(".dng", ".png"))
            imageio.imwrite(output_path, gray_image)

print(f"✅ Saved all grayscale PNGs to: {output_folder}")
