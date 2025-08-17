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
import torchaudio


def process_audio(audio: str | dict, processor=None) -> Tuple[torch.Tensor, int]:
    if isinstance(audio, dict):
        # TODO: to check whether the keys are correct here
        audio_path = audio.get("audio", audio)
    else:
        audio_path = audio

    try:
        # Load audio
        # NOTE: accepts waveform and sample rate; 
        audio_data, original_sr = torchaudio.load(audio_path)

        # Get target sampling rate
        # NOTE: sample rate is basically the amount of audio samples captured per second
        # 16000 means 16000 samples are taken in every second
        if processor and hasattr(processor, 'feature_extractor') and hasattr(processor.feature_extractor,
                                                                             'sampling_rate'):
            target_sr = processor.feature_extractor.sampling_rate
        else:
            
            target_sr = 16000
        print(f"KEANE: Processing audio {audio_path} with sampling rate, {target_sr}")
        # Resample if needed
        # NOTE: This is essentially the resampling of the audio sample rate
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(original_sr, target_sr)
            audio_data = resampler(audio_data)

        # Convert to mono if stereo
        # NOTE: This is essentially the conversion of stereo audio to mono, so that we only have one channel
        if audio_data.shape[0] > 1:
            audio_data = audio_data.mean(dim=0, keepdim=False)
        else:
            audio_data = audio_data.squeeze(0)

        return audio_data, target_sr
    except Exception as e:
        print(f"Error processing audio {audio_path}: {e}")
        dummy_audio = torch.zeros((1000,), dtype=torch.float32)
        return dummy_audio, 16000