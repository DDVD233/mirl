import os
import torch
import numpy as np
import opensmile
from pathlib import Path
import argparse
from typing import Dict, Optional, Union
from tqdm import tqdm
import tempfile
import subprocess
import warnings

# Common audio/video extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', '.wma', '.opus'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def extract_audio_from_video(video_path: Path, temp_audio_path: Path) -> bool:
    """Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to video file
        temp_audio_path: Path where to save extracted audio
    
    Returns:
        True if extraction successful, False otherwise
    """
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', '16000',  # 16kHz sample rate (common for speech)
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(temp_audio_path)
        ]
        
        # Run ffmpeg quietly
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_opensmile_features(
    file_path: Path,
    feature_set: str = 'ComParE_2016',
    feature_level: str = 'functionals'
) -> Dict[str, Union[torch.Tensor, str, int]]:
    """Extract OpenSMILE features from an audio or video file.
    
    Args:
        file_path: Path to audio/video file
        feature_set: OpenSMILE feature set to use
        feature_level: Feature level ('functionals' or 'lld' for low-level descriptors)
    
    Returns:
        Dictionary containing:
        - 'features': tensor of shape (num_windows, num_features) for lld or (1, num_features) for functionals
        - 'feature_names': list of feature names
        - 'file_path': original file path
        - 'feature_set': feature set used
        - 'feature_level': feature level used
    """
    
    # Initialize OpenSMILE
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=feature_level
    )
    
    # Check if input is video or audio
    is_video = file_path.suffix.lower() in VIDEO_EXTENSIONS
    
    if is_video:
        # Extract audio from video first
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio:
            temp_audio_path = Path(temp_audio.name)
            
            if not extract_audio_from_video(file_path, temp_audio_path):
                raise RuntimeError(f"Failed to extract audio from video: {file_path}")
            
            # Process the extracted audio
            features_df = smile.process_file(str(temp_audio_path))
    else:
        # Process audio file directly
        features_df = smile.process_file(str(file_path))
    
    # Convert to numpy array and then to PyTorch tensor
    features_array = features_df.values
    feature_names = features_df.columns.tolist()
    
    # Create output dictionary
    output = {
        'features': torch.tensor(features_array, dtype=torch.float32),
        'feature_names': feature_names,
        'file_path': str(file_path),
        'feature_set': feature_set,
        'feature_level': feature_level,
        'num_windows': features_array.shape[0],
        'num_features': features_array.shape[1]
    }
    
    return output


def process_files_in_directory(
    base_dir: Path,
    feature_set: str = 'ComParE_2016',
    feature_level: str = 'functionals'
):
    """Process all audio/video files in directory and subdirectories, saving features to opensmile folder.
    
    Args:
        base_dir: Base directory to search for files
        feature_set: OpenSMILE feature set to use
        feature_level: Feature level to extract
    """
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Create opensmile directory at the same level as base_dir
    opensmile_dir = base_dir / "opensmile"
    
    print("Scanning for audio and video files...")
    
    # Find all supported files recursively
    supported_files = []
    for ext in SUPPORTED_EXTENSIONS:
        supported_files.extend(list(base_dir.rglob(f"*{ext}")))
        # Also check for uppercase extensions
        supported_files.extend(list(base_dir.rglob(f"*{ext.upper()}")))
    
    # Filter out files already in opensmile directory
    supported_files = [f for f in supported_files if "opensmile" not in f.parts]
    
    # Remove duplicates (in case of case-insensitive file systems)
    supported_files = list(set(supported_files))
    
    if not supported_files:
        print(f"No audio/video files found to process (searched for: {', '.join(SUPPORTED_EXTENSIONS)})")
        return
    
    print(f"Found {len(supported_files)} file(s) to process")
    print(f"Using feature set: {feature_set}")
    print(f"Using feature level: {feature_level}")
    print(f"Features will be saved to: {opensmile_dir}\n")
    
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Check if ffmpeg is available for video processing
    video_files = [f for f in supported_files if f.suffix.lower() in VIDEO_EXTENSIONS]
    if video_files:
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: ffmpeg not found. Video files will be skipped.")
            print("Install ffmpeg to process video files: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            supported_files = [f for f in supported_files if f.suffix.lower() not in VIDEO_EXTENSIONS]
    
    # Main progress bar for all files
    with tqdm(supported_files, desc="Processing files", unit="file") as pbar:
        for file_path in pbar:
            # Get relative path from base_dir
            relative_path = file_path.relative_to(base_dir)
            
            # Create output path maintaining directory structure
            output_path = opensmile_dir / relative_path.with_suffix('.pt')
            
            # Update progress bar description
            file_type = "video" if file_path.suffix.lower() in VIDEO_EXTENSIONS else "audio"
            pbar.set_description(f"Processing {file_type}: {relative_path.name[:30]}")
            
            # Skip if already processed
            if output_path.exists():
                skipped_count += 1
                pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count, 'errors': error_count})
                continue
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Extract features
                features = extract_opensmile_features(
                    file_path,
                    feature_set=feature_set,
                    feature_level=feature_level
                )
                
                # Save features as PyTorch file
                torch.save(features, output_path)
                processed_count += 1
                
                # Update progress bar postfix
                pbar.set_postfix({
                    'processed': processed_count,
                    'skipped': skipped_count,
                    'errors': error_count,
                    'shape': f"{features['features'].shape}"
                })
                
            except Exception as e:
                error_count += 1
                pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count, 'errors': error_count})
                tqdm.write(f"  Error processing {file_path.name}: {e}")
                continue
    
    print(f"\n{'=' * 50}")
    print(f"Processing complete!")
    print(f"✓ Successfully processed: {processed_count} files")
    print(f"⊘ Skipped (already exists): {skipped_count} files")
    print(f"✗ Errors: {error_count} files")
    print(f"Features saved in: {opensmile_dir}")


def load_and_inspect_features(pt_file_path: Path):
    """Helper function to load and inspect saved OpenSMILE features.
    
    Args:
        pt_file_path: Path to .pt file containing features
    """
    
    features = torch.load(pt_file_path)
    
    print(f"\nInspecting features from: {pt_file_path}")
    print(f"Keys: {list(features.keys())}")
    
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, list) and key == 'feature_names':
            print(f"  {key}: {len(value)} features")
            # Show first few feature names as example
            if len(value) > 0:
                print(f"    First 5 features: {value[:5]}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nOriginal file: {features['file_path']}")
    print(f"Feature set: {features['feature_set']}")
    print(f"Feature level: {features['feature_level']}")
    print(f"Number of windows: {features['num_windows']}")
    print(f"Number of features: {features['num_features']}")
    
    # Show statistics for features
    if 'features' in features:
        feat_tensor = features['features']
        print(f"\nFeature statistics:")
        print(f"  Min: {feat_tensor.min().item():.4f}")
        print(f"  Max: {feat_tensor.max().item():.4f}")
        print(f"  Mean: {feat_tensor.mean().item():.4f}")
        print(f"  Std: {feat_tensor.std().item():.4f}")
    
    return features


def list_available_feature_sets():
    """List all available OpenSMILE feature sets."""
    
    print("Available OpenSMILE feature sets:")
    print("=" * 50)
    
    # Common feature sets with descriptions
    feature_sets = {
        'ComParE_2016': 'ComParE 2016 feature set (6373 features)',
        'GeMAPSv01b': 'Geneva Minimalistic Acoustic Parameter Set (62 features)',
        'eGeMAPSv02': 'Extended Geneva Minimalistic Acoustic Parameter Set (88 features)',
        'emobase': 'Emotion recognition baseline set (988 features)',
        'IS09_emotion': 'INTERSPEECH 2009 Emotion Challenge (384 features)',
        'IS10_paraling': 'INTERSPEECH 2010 Paralinguistic Challenge (1582 features)',
        'IS11_speaker_state': 'INTERSPEECH 2011 Speaker State Challenge (4368 features)',
        'IS12_speaker_trait': 'INTERSPEECH 2012 Speaker Trait Challenge (6125 features)',
        'IS13_ComParE': 'INTERSPEECH 2013 ComParE Challenge (6373 features)',
    }
    
    for name, description in feature_sets.items():
        print(f"  {name:20s} - {description}")
    
    print("\nFeature levels:")
    print("  functionals - Statistical functionals over time (default)")
    print("  lld         - Low-level descriptors (frame-by-frame features)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenSMILE features from audio/video files and save as PyTorch tensors"
    )
    parser.add_argument("directory", help="Base directory to search for audio/video files")
    parser.add_argument(
        "--feature-set", 
        default="ComParE_2016",
        help="OpenSMILE feature set to use (default: ComParE_2016)"
    )
    parser.add_argument(
        "--feature-level",
        default="functionals",
        choices=["functionals", "lld"],
        help="Feature level to extract (default: functionals)"
    )
    parser.add_argument(
        "--inspect", 
        action="store_true", 
        help="Inspect existing .pt files instead of processing"
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List available feature sets and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_features:
        list_available_feature_sets()
        return
    
    base_dir = Path(args.directory)
    
    if args.inspect:
        # Inspect mode: look for .pt files in opensmile subdirectory
        opensmile_dir = base_dir / "opensmile"
        
        if not opensmile_dir.exists():
            print(f"No opensmile directory found in {base_dir}")
            return
        
        pt_files = list(opensmile_dir.rglob("*.pt"))
        
        if not pt_files:
            print(f"No .pt files found in {opensmile_dir}")
            return
        
        print(f"Found {len(pt_files)} .pt file(s)")
        print("=" * 50)
        
        # Inspect first few files as examples
        for i, pt_file in enumerate(pt_files[:3]):  # Show first 3 files
            print(f"\nFile {i+1}/{min(3, len(pt_files))}:")
            load_and_inspect_features(pt_file)
            print("-" * 30)
        
        if len(pt_files) > 3:
            print(f"\n... and {len(pt_files) - 3} more files")
    else:
        # Process mode: extract features from files
        process_files_in_directory(
            base_dir,
            feature_set=args.feature_set,
            feature_level=args.feature_level
        )


if __name__ == "__main__":
    main()