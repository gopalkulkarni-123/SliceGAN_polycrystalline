from PIL import Image

# Load your image
img = Image.open('./images/metal_texture.jpg')

# Option A: Resize (Squish) - Not recommended for materials
# resized_img = img.resize((64, 64))

# Option B: Center Crop (Best for maintaining material scale)
width, height = img.size
new_dimension = min(width, height)
left = (width - new_dimension)/2
top = (height - new_dimension)/2
right = (width + new_dimension)/2
bottom = (height + new_dimension)/2

# Crop to square then resize to 64x64
img = img.crop((left, top, right, bottom))
img = img.resize((64, 64), Image.LANCZOS)

img.save('./images/metal_texture_processed.jpg')
print("Image prepped for SliceGAN!")