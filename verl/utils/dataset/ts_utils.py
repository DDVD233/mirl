# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Time-series processing utilities for multimodal datasets."""

import os
from typing import Union, Optional

import torch


def process_time_series(time_series_path: Union[str, torch.Tensor]) -> torch.Tensor:
    """
    Process time-series data from file path or tensor.

    Args:
        time_series_path: Path to time-series file (.pt file) or torch.Tensor

    Returns:
        torch.Tensor: Processed time-series data with shape (channels, sequence_length)
    """
    if isinstance(time_series_path, torch.Tensor):
        return time_series_path.to(torch.float32)

    if isinstance(time_series_path, str):
        if not os.path.exists(time_series_path):
            raise FileNotFoundError(f"Time series file not found: {time_series_path}")

        # Load time-series data from .pt file
        time_series = torch.load(time_series_path)

        # Ensure float32 dtype (convert from bfloat16 if needed)
        if time_series.dtype == torch.bfloat16:
            time_series = time_series.to(torch.float32)

        return time_series

    raise TypeError(f"Unsupported time-series type: {type(time_series_path)}")


def validate_time_series_shape(tensor: torch.Tensor, expected_channels: Optional[int] = 8,
                              expected_length: Optional[int] = 2500) -> bool:
    """
    Validate time-series tensor shape.

    Args:
        tensor: Time-series tensor to validate
        expected_channels: Expected number of channels (default 8 for ECG)
        expected_length: Expected sequence length (default 2500 for ECG)

    Returns:
        bool: True if shape is valid
    """
    if tensor.ndim != 2:
        return False

    channels, length = tensor.shape

    if expected_channels is not None and channels != expected_channels:
        return False

    if expected_length is not None and length != expected_length:
        return False

    return True