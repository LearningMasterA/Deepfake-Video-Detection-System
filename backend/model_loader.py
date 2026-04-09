import torch
from torchvision.models import efficientnet_b0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_load_error = None


def load_model():
    loaded_model = efficientnet_b0(weights=None)

    # Replace classifier for Real/Fake classification.
    loaded_model.classifier[1] = torch.nn.Linear(
        loaded_model.classifier[1].in_features,
        2
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

    classifier_weight = new_state_dict.get("classifier.1.weight")
    legacy_classifier_weight = new_state_dict.get("_fc.weight")

    if classifier_weight is None:
        if legacy_classifier_weight is not None:
            raise RuntimeError(
                "models/deepfake_model.pth uses an incompatible EfficientNet "
                f"checkpoint with {legacy_classifier_weight.shape[0]} output "
                "classes. Replace it with a torchvision EfficientNet-B0 "
                "checkpoint whose final classifier has 2 outputs."
            )

        raise RuntimeError(
            "models/deepfake_model.pth does not contain the binary classifier "
            "weights expected at classifier.1.weight."
        )

    if classifier_weight.shape[0] != 2:
        raise RuntimeError(
            "models/deepfake_model.pth is not the expected Real/Fake deepfake "
            f"classifier. Expected 2 output classes, found {classifier_weight.shape[0]}."
        )

    # Load strictly so incompatible checkpoints cannot produce random predictions.
    loaded_model.load_state_dict(new_state_dict, strict=True)

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


try:
    model = load_model()
except Exception as exc:
    model_load_error = str(exc)
