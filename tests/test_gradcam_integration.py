import numpy as np
import torch
import torch.nn as nn

from deepfake_detection.gradcam import GradCAM, overlay_heatmap
from deepfake_detection.preprocessing import make_default_preprocess
from deepfake_detection.video_pipeline import DeepfakeVideoExplainer, ExplanationResult


class TinyCNN(nn.Module):
    def __init__(self, out_dim=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_gradcam_map_shape_and_range_multiclass():
    model = TinyCNN(out_dim=2).eval()
    gradcam = GradCAM(model, model.features[2], method="gradcam")

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    cam = gradcam(x)

    assert cam.shape == (2, 32, 32)
    assert torch.all(cam >= 0)
    assert torch.all(cam <= 1)


def test_gradcam_plus_plus_works():
    model = TinyCNN(out_dim=2).eval()
    gradcam = GradCAM(model, model.features[2], method="gradcam++")

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    cam = gradcam(x)
    assert cam.shape == (2, 32, 32)


def test_gradcam_handles_single_logit_binary_head_for_both_classes():
    model = TinyCNN(out_dim=1).eval()
    gradcam = GradCAM(model, model.features[2])

    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    cam_fake = gradcam(x, target_class=1)
    cam_real = gradcam(x, target_class=0)

    assert cam_fake.shape == (1, 32, 32)
    assert cam_real.shape == (1, 32, 32)


def test_overlay_heatmap_returns_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cam_map = np.random.rand(64, 64).astype(np.float32)

    out = overlay_heatmap(img, cam_map, blur_kernel=7)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_default_preprocess_output_shape():
    preprocess = make_default_preprocess(image_size=128)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tensor = preprocess(frame)

    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == torch.float32


def test_probability_from_logits_respects_fake_class_index():
    model = TinyCNN(out_dim=2).eval()
    explainer = DeepfakeVideoExplainer(
        model=model,
        target_layer=model.features[2],
        preprocess=make_default_preprocess(),
        fake_class_index=0,
        enable_tta=False,
    )

    probs = explainer.probability_from_logits(torch.tensor([[2.0, 1.0]]))
    assert probs.shape == (1,)
    assert float(probs[0]) > 0.5


def test_explanation_result_dataclass_fields():
    result = ExplanationResult(
        fake_score=0.7,
        predicted_label="FAKE",
        decision_threshold=0.55,
        sampled_frames=10,
        total_frames=50,
        heatmap_video_path=None,
    )
    assert result.predicted_label == "FAKE"
    assert result.fake_score >= result.decision_threshold
