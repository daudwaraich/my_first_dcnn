import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

folder = "dataset/animals"

for filename in os.listdir(folder):
    if filename.lower().endswith(".heic"):
        old_path = os.path.join(folder, filename)
        new_filename = filename[:-5] + ".jpg"
        new_path = os.path.join(folder, new_filename)
        
        img = Image.open(old_path)
        img.save(new_path, "JPEG")
        os.remove(old_path)
        print(f"Converted: {filename} → {new_filename}")

print("All HEIC files converted!")