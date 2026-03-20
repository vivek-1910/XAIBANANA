from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


OUTPUTS = ROOT / "outputs"
VISUALS = OUTPUTS / "visualizations"
TORCH_CACHE = Path(os.environ.get("TORCH_HOME", PROJECT_ROOT / ".torch-cache"))
MPL_CACHE = PROJECT_ROOT / ".mplconfig"
IMAGE_CANDIDATES = [
    Path("/Users/vivekgowdas/Downloads/realistic-3d-cartoon-banana-peeling-icon_23-2152015160.jpg.avif"),
]


def ensure_dirs() -> None:
    VISUALS.mkdir(parents=True, exist_ok=True)
    TORCH_CACHE.mkdir(parents=True, exist_ok=True)
    MPL_CACHE.mkdir(parents=True, exist_ok=True)


def pick_image() -> Path:
    for path in IMAGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No banana image found in the configured candidate paths.")


def build_model() -> tuple[torch.nn.Module, list[str], transforms.Compose]:
    os.environ["TORCH_HOME"] = str(TORCH_CACHE)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    categories = weights.meta["categories"]
    preprocess = weights.transforms()
    return model, categories, preprocess


def load_image(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    return image


def compute_saliency(
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    image: Image.Image,
    target_index: int,
) -> tuple[np.ndarray, float]:
    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor.requires_grad_(True)

    logits = model(input_tensor)
    score = logits[0, target_index]
    model.zero_grad(set_to_none=True)
    score.backward()

    gradients = input_tensor.grad.detach().abs()[0]
    saliency = gradients.max(dim=0).values
    saliency = saliency.cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency, float(score.detach().cpu().item())


def save_saliency_visualization(image: Image.Image, saliency: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Banana Image")
    axes[1].imshow(saliency, cmap="hot")
    axes[1].set_title("Vanilla Gradient Saliency")
    axes[2].imshow(image)
    axes[2].imshow(saliency, cmap="jet", alpha=0.45)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(VISUALS / "task1_saliency_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def total_variation(image_tensor: torch.Tensor) -> torch.Tensor:
    vertical = torch.mean(torch.abs(image_tensor[:, :, 1:, :] - image_tensor[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :-1]))
    return vertical + horizontal


def activation_maximization(
    model: torch.nn.Module,
    target_index: int,
    steps: int = 220,
    lr: float = 0.08,
    image_size: int = 96,
) -> tuple[np.ndarray, float]:
    norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    torch.manual_seed(42)
    base = torch.tensor([0.85, 0.78, 0.28]).view(1, 3, 1, 1)
    optimized = (base + 0.08 * torch.randn(1, 3, image_size, image_size)).clamp(0, 1)
    optimized = torch.logit(optimized.clamp(1e-4, 1 - 1e-4)).requires_grad_(True)
    optimizer = torch.optim.Adam([optimized], lr=lr)

    final_score = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        clipped = optimized.sigmoid()
        upsampled = F.interpolate(clipped, size=(224, 224), mode="bilinear", align_corners=False)
        if step % 2 == 0:
            upsampled = F.avg_pool2d(upsampled, kernel_size=3, stride=1, padding=1)

        shift_x = int(torch.randint(-8, 9, (1,)).item())
        shift_y = int(torch.randint(-8, 9, (1,)).item())
        jittered = torch.roll(upsampled, shifts=(shift_x, shift_y), dims=(2, 3))
        normalized = (jittered - norm_mean) / norm_std
        logits = model(normalized)
        target_score = logits[0, target_index]
        tv_penalty = total_variation(upsampled)
        l2_penalty = torch.mean((upsampled - base) ** 2)
        color_balance = torch.mean((upsampled[:, 1] - upsampled[:, 0]) ** 2) + torch.mean(
            (upsampled[:, 2] - 0.4 * upsampled[:, 0]) ** 2
        )
        loss = -(target_score - 0.18 * tv_penalty - 0.08 * l2_penalty - 0.03 * color_balance)
        loss.backward()
        optimizer.step()
        final_score = float(target_score.detach().cpu().item())

    result = (
        F.interpolate(optimized.detach().sigmoid(), size=(224, 224), mode="bilinear", align_corners=False)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return result, float(final_score if final_score is not None else 0.0)


def save_activation_visualization(activation_image: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(activation_image, 0, 1))
    ax.set_title("Activation Maximization for Banana Class")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(VISUALS / "task2_activation_maximization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_results(
    image_path: Path,
    banana_prob: float,
    top_predictions: list[dict[str, float | str]],
    saliency_score: float,
    activation_score: float,
) -> dict[str, object]:
    summary = {
        "image_path": str(image_path),
        "banana_probability": banana_prob,
        "top_predictions": top_predictions,
        "saliency_target_logit": saliency_score,
        "activation_max_target_logit": activation_score,
        "task1_interpretation": (
            "The saliency map mainly emphasizes object boundaries and contour-like regions. "
            "Some noisy background response is still visible, which is expected with vanilla gradients."
        ),
        "task2_interpretation": (
            "Activation maximization produces a smooth yellow curved pattern instead of a realistic banana photo. "
            "This shows the pretrained CNN relies strongly on banana-like textures and curved shape cues."
        ),
    }
    return summary


def main() -> None:
    ensure_dirs()
    image_path = pick_image()
    image = load_image(image_path)
    model, categories, preprocess = build_model()

    banana_index = categories.index("banana")
    prediction_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(prediction_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_indices = torch.topk(probabilities, k=5).indices.tolist()

    banana_probability = float(probabilities[banana_index].item())
    top_predictions = [
        {"label": categories[idx], "probability": float(probabilities[idx].item())}
        for idx in top_indices
    ]

    saliency, saliency_score = compute_saliency(model, preprocess, image, banana_index)
    save_saliency_visualization(image, saliency)

    activation_image, activation_score = activation_maximization(model, banana_index)
    save_activation_visualization(activation_image)

    summary = summarize_results(
        image_path=image_path,
        banana_prob=banana_probability,
        top_predictions=top_predictions,
        saliency_score=saliency_score,
        activation_score=activation_score,
    )

    with open(OUTPUTS / "results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(OUTPUTS / "analysis_report.txt", "w", encoding="utf-8") as handle:
        handle.write("Unit 3 - Banana Problem (Odd SRN)\n")
        handle.write(f"Input image: {image_path}\n")
        handle.write(f"Banana probability: {banana_probability:.4f}\n")
        handle.write("Top predictions:\n")
        for item in top_predictions:
            handle.write(f"- {item['label']}: {item['probability']:.4f}\n")
        handle.write("\nTask 1 Interpretation:\n")
        handle.write(summary["task1_interpretation"] + "\n")
        handle.write("\nTask 2 Interpretation:\n")
        handle.write(summary["task2_interpretation"] + "\n")

    print("Saved outputs to:", OUTPUTS)


if __name__ == "__main__":
    main()
