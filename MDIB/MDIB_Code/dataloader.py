import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def populate_train_list(lowlight_images_path: str) -> List[str]:
    """Return a shuffled list of training image paths."""
    random.seed(1143)
    image_list = sorted(Path(lowlight_images_path).glob("*.jpg"))
    random.shuffle(image_list)
    return [str(p) for p in image_list]


class LowLightLoader(data.Dataset):
    """Dataset for low-light images, resized to a fixed square resolution."""

    def __init__(self, lowlight_images_path: str, size: int = 256) -> None:
        self.size = size
        self.data_list = populate_train_list(lowlight_images_path)
        print("Total training examples:", len(self.data_list))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = Path(self.data_list[index])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size, self.size), Image.Resampling.LANCZOS)

        arr = (np.asarray(img, dtype=np.float32) / 255.0)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor