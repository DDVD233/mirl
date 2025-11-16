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

from typing import Tuple, Union
import torch
import torchaudio

def process_audio(
    audio: Union[str, dict],
    processor=None,
    max_seconds: float = 10  # keep audio to this many seconds max
) -> Tuple[torch.Tensor, int]:
    """
    Load audio, convert to mono, resample, and clip to max_seconds.
    """
    if isinstance(audio, dict):
        audio_path = audio.get("audio", audio)
    else:
        audio_path = audio

    try:
        # Load
        audio_data, original_sr = torchaudio.load(audio_path)

        # Resample if needed
        if processor and hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor, 'sampling_rate'):
            target_sr = processor.feature_extractor.sampling_rate
        else:
            target_sr = 16000

        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(original_sr, target_sr)
            audio_data = resampler(audio_data)
        else:
            target_sr = original_sr

        # Convert to mono
        if audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0, keepdim=False)
        else:
            audio_data = audio_data.squeeze(0)

        # Clip to max_seconds
        if max_seconds:
            max_samples = int(max_seconds * target_sr)
            
            # print(f"Processing Audio {audio_path}, shape={audio_data.shape}, "
            #         f"sr={target_sr}, max_samples={max_samples}")
            # ValueError("Audio was processed")

            if audio_data.shape[0] > max_samples:
                print("Clipping audio to max_seconds")
                audio_data = audio_data[:max_samples]

        return audio_data, target_sr

    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        print("Appending dummy seconds")
        dummy_seconds = 0.5
        dummy_audio = torch.zeros((int(16000 * dummy_seconds),), dtype=torch.float32)
        return dummy_audio, 16000


# def process_audio(
#     audio: Union[str, dict],
#     processor=None,
#     max_seconds: float = 10.0  # uniform duration in seconds
# ) -> Tuple[torch.Tensor, int]:
#     """
#     Load audio, convert to mono, resample, clip or pad to exactly max_seconds.
#     Returns:
#         audio_data (torch.Tensor): Tensor of shape [num_samples]
#         target_sr (int): Sampling rate used
#     """
#     # Determine target sampling rate
#     if processor and hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
#         target_sr = processor.feature_extractor.sampling_rate
#     else:
#         target_sr = 16000

#     # Expected number of samples
#     max_samples = int(max_seconds * target_sr)

#     # Get path if audio is a dict
#     if isinstance(audio, dict):
#         audio_path = audio.get("audio", audio)
#     else:
#         audio_path = audio

#     try:
#         # Load and resample if needed
#         audio_data, original_sr = torchaudio.load(audio_path)

#         if original_sr != target_sr:
#             resampler = torchaudio.transforms.Resample(original_sr, target_sr)
#             audio_data = resampler(audio_data)

#         # Convert to mono
#         if audio_data.shape[0] > 1:
#             audio_data = audio_data.mean(dim=0)
#         else:
#             audio_data = audio_data.squeeze(0)

#         # Clip or pad to uniform length
#         num_samples = audio_data.shape[0]
#         if num_samples > max_samples:
#             audio_data = audio_data[:max_samples]
#         elif num_samples < max_samples:
#             pad_length = max_samples - num_samples
#             audio_data = torch.nn.functional.pad(audio_data, (0, pad_length))

#         return audio_data, target_sr

#     except Exception as e:
#         print(f"[WARN] Error processing audio {audio_path}: {e}")
#         # Create uniform dummy audio (silence)
#         dummy_audio = torch.zeros((max_samples,), dtype=torch.float32)
#         return dummy_audio, target_sr