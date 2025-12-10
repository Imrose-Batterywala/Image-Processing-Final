import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
import torchvision
from PIL import Image

import dataloader  # noqa: F401  (kept for compatibility with original code)
import model


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    net = model.enhance_net_nopool().to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.eval()
    return net


def prepare_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """Load and normalize image to tensor in CHW format."""
    img = Image.open(image_path).convert("RGB")
    arr = (np.asarray(img) / 255.0).astype(np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def run_inference(net: torch.nn.Module, image_path: Path, device: torch.device) -> torch.Tensor:
    data = prepare_image(image_path, device)
    with torch.no_grad():
        enhanced, _ = net(data)
    return enhanced


def save_output(enhanced: torch.Tensor, image_path: Path, data_root: Path, output_root: Path) -> None:
    """Save enhanced image preserving dataset/subfolder structure."""
    relative = image_path.relative_to(data_root)
    save_path = output_root / relative
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(enhanced, save_path)
    print(f"saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Low-light enhancement inference (MDIB)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/test_data",
        help="Root directory with test images (expects subfolders).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/result",
        help="Where to write enhanced images (mirrors input structure).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="snapshots/Epoch99.pth",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    data_root = Path(args.data_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    weights_path = Path(args.weights).resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.backends.cudnn.benchmark = True

    net = load_model(weights_path, device)

    start = time.time()
    for image_path in data_root.rglob("*"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        enhanced = run_inference(net, image_path, device)
        save_output(enhanced, image_path, data_root, output_root)
    total = time.time() - start
    print(f"Done in {total:.2f}s")


if __name__ == "__main__":
    main()