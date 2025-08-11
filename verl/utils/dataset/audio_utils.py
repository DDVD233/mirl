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

from typing import Tuple, Optional

import torch
from transformers.pipelines.audio_utils import ffmpeg_read


def process_audio(audio: str | dict, processor=None) -> Tuple[torch.Tensor, int]:
    """Process audio file and return (audio_data, sampling_rate) tuple.
    
    Args:
        audio: Audio file path (str) or audio dict containing file path
        processor: Audio processor with feature_extractor for sampling rate
        
    Returns:
        Tuple of (audio_data, sampling_rate)
    """
    if isinstance(audio, dict):
        audio_path = audio.get("audio", audio)
    else:
        audio_path = audio
    
    try:
        # Get sampling rate from processor if available, otherwise use default
        if processor and hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
            sampling_rate = processor.feature_extractor.sampling_rate
        else:
            sampling_rate = 16000  # Default sampling rate
            
        # Read audio using ffmpeg_read with the specified sampling rate
        audio_data = ffmpeg_read(audio_path, sampling_rate=sampling_rate)
        
        return audio_data, sampling_rate
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        # Return dummy audio data on error
        dummy_audio = torch.zeros((1000,), dtype=torch.float32)  # 1 second of silence at 1kHz
        return dummy_audio, 16000