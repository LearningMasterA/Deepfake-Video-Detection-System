import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    model.eval()
    return model

model = load_model()