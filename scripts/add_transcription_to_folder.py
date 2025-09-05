#!/usr/bin/env python3
"""
Add Whisper transcriptions to audio/video files in a folder structure.
Saves transcriptions as SRT files with timestamps.
"""

import os
import sys
import torch
import whisper
from pathlib import Path
import argparse
from tqdm import tqdm

# Supported audio/video extensions
SUPPORTED_EXTENSIONS = {
    '.mp3', '.mp4', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.opus',
    '.avi', '.mov', '.mkv', '.webm', '.mpeg', '.mpg', '.wmv', '.flv'
}


def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, output_path):
    """Write segments to SRT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n")
            f.write("\n")


def process_file(file_path, model, input_folder):
    """Process a single audio/video file"""
    file_path = Path(file_path)
    input_folder = Path(input_folder)
    
    # Create output path in transcription subfolder, maintaining structure
    relative_path = file_path.relative_to(input_folder)
    output_path = input_folder / "transcription" / relative_path.with_suffix('.srt')
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if already processed
    if output_path.exists():
        print(f"Skipping {file_path.name} (SRT already exists)")
        return
    
    print(f"Processing: {file_path.name}")
    
    # Transcribe with Whisper
    try:
        result = model.transcribe(
            str(file_path),
            language=None,  # Auto-detect language
            task="transcribe",
            verbose=False
        )
    
        # Write SRT file
        write_srt(result['segments'], output_path)
        print(f"  -> Saved: {output_path.relative_to(input_folder)}")
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")


def process_folder(input_folder, model=None):
    """Process all audio/video files in folder and subfolders"""
    input_path = Path(input_folder)
    
    # Find all supported files
    files_to_process = []
    for ext in SUPPORTED_EXTENSIONS:
        files_to_process.extend(input_path.rglob(f'*{ext}'))
        files_to_process.extend(input_path.rglob(f'*{ext.upper()}'))
    
    if not files_to_process:
        print(f"No audio/video files found in {input_folder}")
        return
    
    print(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    for file_path in tqdm(files_to_process, desc="Transcribing"):
        process_file(file_path, model, input_path)


def main():
    parser = argparse.ArgumentParser(
        description='Add Whisper transcriptions to audio/video files'
    )
    parser.add_argument(
        'input_folder',
        help='Input folder containing audio/video files'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--model',
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Whisper model size (default: large-v3)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Loading Whisper {args.model} model on {args.device}...")
    model = whisper.load_model(args.model, device=args.device)
    
    # Process folder
    process_folder(args.input_folder, model)
    
    print("Done!")


if __name__ == "__main__":
    main()