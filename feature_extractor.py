import torch
import torchvision
from torchvision.models import ResNet101_Weights 
from torchvision import transforms
from PIL import Image
# load resnet101 model and remove the last layer
resnet101 = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT)
resnet101 = torch.nn.Sequential(*(list(resnet101.children())[:-1]))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_resnet101_features(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet101(img)

    return features.squeeze().numpy()

# img_url = '/content/chuot.webp'
# image_features = extract_resnet101_features(img_url)
# image_features