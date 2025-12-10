from typing import Tuple

import torch
import torch.nn as nn


class enhance_net_nopool(nn.Module):
    """Lightweight enhancement network without pooling."""

    def __init__(self) -> None:
        super().__init__()
        features = 32

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, features, 3, stride=1, padding=1, bias=True)
        self.e_conv2 = nn.Conv2d(features, features, 3, stride=1, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(features, features, 3, stride=1, padding=1, bias=True)
        self.e_conv7 = nn.Conv2d(features * 2, 3, 3, stride=1, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Original branch
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x3], dim=1)))

        # Inverted branch
        xx = 1 - x
        x11 = self.relu(self.e_conv1(xx))
        x21 = self.relu(self.e_conv2(x11))
        x31 = self.relu(self.e_conv3(x21))
        x_r1 = torch.tanh(self.e_conv7(torch.cat([x11, x31], dim=1)))

        # Blend both adjustment maps
        x_r = (x_r + x_r1) / 2

        # Adaptive iteration count
        mean_x = torch.mean(x).item()
        n1 = 0.63
        s = mean_x * mean_x
        n3 = -0.79 * s + 0.81 * mean_x + 1.4

        if mean_x < 0.1:
            b = -25 * mean_x + 10
        elif mean_x < 0.45:
            b = 17.14 * s - 15.14 * mean_x + 10
        else:
            b = 5.66 * s - 2.93 * mean_x + 7.2
        b = int(b)

        for _ in range(b):
            mean_current = torch.mean(x).item()
            x = x + x_r * (torch.pow(x, 2) - x) * ((n1 - mean_current) / (n3 - mean_current))

        enhance_image = x
        return enhance_image, x_r
