import torch
import torchvision.models as models

# Load ResNet50 with pre-trained weights
resnet50 = models.resnet50(pretrained=True)

# Set the model to evaluation mode (not necessary if you want to fine-tune)
resnet50.eval()
