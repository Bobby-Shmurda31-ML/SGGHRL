from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImageVectorExtractor(BaseFeaturesExtractor):
    """
    Features extractor for Dict spaces: {"image": Box(C,H,W), "vector": Box(D,)}.
    Works with ANY image size (including < 36×36) thanks to AdaptiveAvgPool2d.

    Usage with SB3::

        policy_kwargs = {
            "features_extractor_class": ImageVectorExtractor,
            "features_extractor_kwargs": {
                "features_dim": 64,
                "cnn_channels": [16, 32],
            },
        }
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64,
        cnn_channels: Optional[List[int]] = None,
        image_key: str = "image",
        vector_key: str = "vector",
    ):
        super().__init__(observation_space, features_dim)

        if cnn_channels is None:
            cnn_channels = [16, 32]

        assert image_key in observation_space.spaces, (
            f"ImageVectorExtractor requires '{image_key}' key in observation_space, "
            f"got keys: {list(observation_space.spaces.keys())}"
        )

        image_space = observation_space[image_key]
        in_channels = image_space.shape[0]

        layers: list = []
        ch = in_channels
        for out_ch in cnn_channels:
            layers.extend([nn.Conv2d(ch, out_ch, 3, stride=1, padding=1), nn.ReLU()])
            ch = out_ch
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        cnn_out_dim = cnn_channels[-1] if cnn_channels else in_channels

        self._has_vector = vector_key in observation_space.spaces
        vector_dim = observation_space[vector_key].shape[0] if self._has_vector else 0

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + vector_dim, features_dim),
            nn.ReLU(),
        )

        self._image_key = image_key
        self._vector_key = vector_key
        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        cnn_out = self.cnn(observations[self._image_key])

        if self._has_vector:
            combined = torch.cat([cnn_out, observations[self._vector_key]], dim=1)
        else:
            combined = cnn_out

        return self.fc(combined)