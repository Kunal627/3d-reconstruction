import os
from glob import glob
from PIL import Image
import pillow_heif

# the code converts HEIC images to png
# Register HEIF opener with PIL
pillow_heif.register_heif_opener()

#input_folder = "./camera_calibration/images_heic"
#output_folder = "./camera_calibration/images_png"

input_folder = "./stereo_vision/images_heic"
output_folder = "./stereo_vision/images_png"
os.makedirs(output_folder, exist_ok=True)

heic_files = glob(os.path.join(input_folder, "*.HEIC")) + \
             glob(os.path.join(input_folder, "*.heic"))

print("Found", len(heic_files), "HEIC files.")

for path in heic_files:
    img = Image.open(path)

    # Output filename
    base = os.path.basename(path)
    new_name = os.path.splitext(base)[0] + ".png"
    out_path = os.path.join(output_folder, new_name)

    img.save(out_path, "PNG")
    print("Converted:", path, "→", out_path)

print("Done!")
