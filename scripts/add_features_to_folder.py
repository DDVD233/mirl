import os
import cv2
import mediapipe as mp
import torch
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands


def extract_features_from_video(video_path: Path) -> Dict[str, torch.Tensor]:
    """Extract pose, face, and hand features from a video file.

    Returns:
        Dictionary containing:
        - 'pose': tensor of shape (num_frames, 33, 3) for pose landmarks
        - 'face': tensor of shape (num_frames, 468, 3) for face landmarks
        - 'left_hand': tensor of shape (num_frames, 21, 3) for left hand landmarks
        - 'right_hand': tensor of shape (num_frames, 21, 3) for right hand landmarks
        - 'frame_indices': tensor of valid frame indices
    """

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize feature lists
    pose_features = []
    face_features = []
    left_hand_features = []
    right_hand_features = []
    valid_frames = []

    # Initialize MediaPipe models
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose, \
            mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh, \
            mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

        # Create progress bar for frames
        with tqdm(total=total_frames, desc=f"  Extracting features", unit="frames", leave=False) as pbar:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # Process pose, face, and hands
                pose_results = pose.process(image_rgb)
                face_results = face_mesh.process(image_rgb)
                hand_results = hands.process(image_rgb)

                # Extract pose landmarks (33 landmarks x 3 coordinates)
                if pose_results.pose_landmarks:
                    pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark])
                else:
                    pose_landmarks = np.zeros((33, 3))
                pose_features.append(pose_landmarks)

                # Extract face landmarks (468 landmarks x 3 coordinates)
                if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
                    face_landmarks = np.array(
                        [[lm.x, lm.y, lm.z] for lm in face_results.multi_face_landmarks[0].landmark])
                else:
                    face_landmarks = np.zeros((468, 3))
                face_features.append(face_landmarks)

                # Extract hand landmarks (21 landmarks x 3 coordinates per hand)
                left_hand = np.zeros((21, 3))
                right_hand = np.zeros((21, 3))

                if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                    for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                                          hand_results.multi_handedness):
                        hand_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        if handedness.classification[0].label == 'Left':
                            left_hand = hand_array
                        else:
                            right_hand = hand_array

                left_hand_features.append(left_hand)
                right_hand_features.append(right_hand)
                valid_frames.append(frame_count - 1)  # 0-indexed

                # Update progress bar
                pbar.update(1)
                if frame_count % 100 == 0:
                    pbar.set_postfix({'fps': f'{fps:.1f}', 'detected': f'{len(valid_frames)}'})

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Convert to PyTorch tensors with proper handling for variable sizes
    # Stack arrays safely, handling potential shape mismatches
    def safe_stack(feature_list, expected_shape):
        """Safely stack feature arrays, padding if necessary."""
        if not feature_list:
            return torch.zeros((0, *expected_shape), dtype=torch.float32)
        
        # Check if all arrays have the same shape
        shapes = [arr.shape for arr in feature_list]
        if len(set(shapes)) == 1:
            # All shapes are the same, can stack normally
            return torch.tensor(np.array(feature_list), dtype=torch.float32)
        else:
            # Shapes differ, need to pad
            max_shape = [len(feature_list)]
            for dim in range(len(expected_shape)):
                max_shape.append(max(arr.shape[dim] if dim < len(arr.shape) else expected_shape[dim] 
                                   for arr in feature_list))
            
            # Create padded array
            padded = np.zeros(max_shape, dtype=np.float32)
            for i, arr in enumerate(feature_list):
                if arr.ndim == len(expected_shape):
                    slices = tuple(slice(None, s) for s in arr.shape)
                    padded[i][slices] = arr
                else:
                    # Handle case where dimensions don't match expected
                    padded[i][:arr.shape[0] if arr.ndim > 0 else 0] = arr.flatten()[:padded.shape[1]]
            
            return torch.tensor(padded, dtype=torch.float32)
    
    features = {
        'pose': safe_stack(pose_features, (33, 3)),
        'face': safe_stack(face_features, (468, 3)),
        'left_hand': safe_stack(left_hand_features, (21, 3)),
        'right_hand': safe_stack(right_hand_features, (21, 3)),
        'frame_indices': torch.tensor(valid_frames, dtype=torch.long),
        'total_frames': total_frames,
        'video_path': str(video_path)
    }

    return features


def process_videos_in_directory(base_dir: Path):
    """Process all videos in directory and subdirectories, saving features to pose folder."""

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return

    # Create pose directory at the same level as base_dir
    pose_dir = base_dir / "pose"

    print("Scanning for video files...")
    # Find all .mp4 files recursively
    mp4_files = list(base_dir.rglob("*.mp4"))

    # Filter out files already in pose directory
    mp4_files = [f for f in mp4_files if "pose" not in f.parts]

    if not mp4_files:
        print("No .mp4 files found to process")
        return

    print(f"Found {len(mp4_files)} .mp4 file(s) to process")
    print(f"Features will be saved to: {pose_dir}\n")

    processed_count = 0
    error_count = 0
    skipped_count = 0

    # Main progress bar for all videos
    with tqdm(mp4_files, desc="Processing videos", unit="video") as pbar:
        for mp4_file in pbar:
            # Get relative path from base_dir
            relative_path = mp4_file.relative_to(base_dir)

            # Create output path maintaining directory structure
            output_path = pose_dir / relative_path.with_suffix('.pt')

            # Update progress bar description
            pbar.set_description(f"Processing: {relative_path.name[:30]}")

            # Skip if already processed
            if output_path.exists():
                skipped_count += 1
                pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count, 'errors': error_count})
                continue

            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Extract features
                features = extract_features_from_video(mp4_file)

                # Save features as PyTorch file
                torch.save(features, output_path)
                processed_count += 1

                # Update progress bar postfix
                pbar.set_postfix({
                    'processed': processed_count,
                    'skipped': skipped_count,
                    'errors': error_count,
                    'shapes': f"P:{features['pose'].shape[0]}x33"
                })

            except Exception as e:
                error_count += 1
                pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count, 'errors': error_count})
                tqdm.write(f"  Error processing {mp4_file.name}: {e}")
                continue

    print(f"\n{'=' * 50}")
    print(f"Processing complete!")
    print(f"✓ Successfully processed: {processed_count} videos")
    print(f"⊘ Skipped (already exists): {skipped_count} videos")
    print(f"✗ Errors: {error_count} videos")
    print(f"Features saved in: {pose_dir}")


def load_and_inspect_features(pt_file_path: Path):
    """Helper function to load and inspect saved features."""

    features = torch.load(pt_file_path)

    print(f"\nInspecting features from: {pt_file_path}")
    # print features keys and variable types
    print(f"Keys: {list(features.keys())}")
    for key, value in features.items():
        print(f"  {key}: type={type(value)}, shape={value.shape if isinstance(value, torch.Tensor) else 'N/A'}")

    print(f"Original video: {features['video_path']}")
    print(f"Total frames in video: {features['total_frames']}")
    print(f"Processed frames: {len(features['frame_indices'])}")
    print(f"Pose landmarks shape: {features['pose'].shape}")
    print(f"Face landmarks shape: {features['face'].shape}")
    print(f"Left hand shape: {features['left_hand'].shape}")
    print(f"Right hand shape: {features['right_hand'].shape}")

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe features from videos and save as PyTorch tensors")
    parser.add_argument("directory", help="Base directory to search for .mp4 files")
    parser.add_argument("--inspect", action="store_true", help="Inspect existing .pt files instead of processing")
    args = parser.parse_args()

    base_dir = Path(args.directory)
    
    if args.inspect:
        # Inspect mode: look for .pt files in base directory and optionally in pose subdirectory
        search_dirs = [base_dir]
        pose_dir = base_dir / "pose"
        if pose_dir.exists():
            search_dirs.append(pose_dir)
        
        pt_files = []
        for search_dir in search_dirs:
            pt_files.extend(list(search_dir.rglob("*.pt")))
        
        # Remove duplicates if any
        pt_files = list(set(pt_files))
        
        if not pt_files:
            print(f"No .pt files found in {base_dir}")
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
        # Process mode: extract features from videos
        process_videos_in_directory(base_dir)


if __name__ == "__main__":
    main()