import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model architecture
model = efficientnet_b0(weights=None)

# Replace classifier for binary classification
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    1
)

# Load checkpoint
checkpoint = torch.load(
    "models/deepfake_model.pth",
    map_location=device,
    weights_only=False
)

# Some checkpoints store weights under "state_dict"
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Remove "module." prefix if model was trained using DataParallel
new_state_dict = {}
for k, v in checkpoint.items():
    if k.startswith("module."):
        k = k[7:]
    new_state_dict[k] = v

# Load weights with strict=False to ignore unmatched layers
model.load_state_dict(new_state_dict, strict=False)

model = model.to(device)
model.eval()