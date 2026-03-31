import os
from torchvision import transforms
from PIL import Image

# Loads images from the data/img folder and returns them as a list of PIL images
def load_as_PIL():
    images = []
    for filename in os.listdir("./data/img"):
        img_path = os.path.join("./data/img", filename)
        img = Image.open(img_path)
        # Do something with the image, e.g. display it
        images.append(img)
    return images

# Convert images to 4d RGBW tensor for cnn input
def image_to_rgbw_tensor():
    img = Image.open('path/to/your/image.jpg')

    transform = transforms.ToTensor()
    img_tensor_3d = transform(img)
    img_tensor_4d = img_tensor_3d.unsqueeze(0)
    
    print(f"Final 4D tensor shape for CNN input: {img_tensor_4d.shape}")
    return img_tensor_4d

# Process the data as needed (e.g., normalization, augmentation, etc.)
def process_images():
    # TODO: Implement any additional processing steps such as normalization, data augmentation, etc.
    pass

# Process the landmarking results to prepare them for nn model training
def landmarking_to_tensor():
    # TODO: Implement landmarking + processing into format for nn model 
    pass
