from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Match training input size
    transforms.ToTensor(),
])

def preprocess_image(file_storage):
    image = Image.open(file_storage).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor