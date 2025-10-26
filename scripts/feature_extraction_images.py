import cv2
import mediapipe as mp
import os
import argparse
from pathlib import Path

# Initialize MediaPipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def process_image(image_path, output_path):
    """
    Process a single image to extract and overlay facial and body keypoints.

    Args:
        image_path: Path to the input image
        output_path: Path where the processed image will be saved
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return False

    # Convert BGR to RGB (MediaPipe uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a copy for drawing
    annotated_image = image.copy()

    # Process with Face Mesh
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,  # Detect up to 10 faces
            refine_landmarks=True,
            min_detection_confidence=0.7
    ) as face_mesh:
        results_face = face_mesh.process(image_rgb)

        # Draw face landmarks
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # Draw face mesh tesselation
                mp_drawing.draw_landmarks(
                    annotated_image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Draw face mesh contours
                mp_drawing.draw_landmarks(
                    annotated_image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                # Draw irises
                mp_drawing.draw_landmarks(
                    annotated_image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

    # Process with Pose
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7
    ) as pose:
        results_pose = pose.process(image_rgb)

        # Draw pose landmarks
        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

    # Process with Hands
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
    ) as hands:
        results_hands = hands.process(image_rgb)

        # Draw hand landmarks
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

    # Save the annotated image
    success = cv2.imwrite(str(output_path), annotated_image)
    if success:
        print(f"Processed: {image_path.name} -> {output_path.name}")
    else:
        print(f"Error: Could not save image {output_path}")

    return success


def process_folder(input_folder, output_folder=None):
    """
    Process all images in a folder.

    Args:
        input_folder: Path to the input folder containing images
        output_folder: Path to the output folder (optional, defaults to input_folder + '_keypoints')
    """
    input_path = Path(input_folder)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: {input_folder} does not exist or is not a directory")
        return

    # Set default output folder if not provided
    if output_folder is None:
        output_path = input_path.parent / f"{input_path.name}_keypoints"
    else:
        output_path = Path(output_folder)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # Get all image files
    image_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Output folder: {output_path}")
    print("-" * 50)

    # Process each image
    successful = 0
    failed = 0

    for image_file in image_files:
        # Create output filename with _keypoints suffix
        output_file = output_path / f"{image_file.stem}_keypoints{image_file.suffix}"

        if process_image(image_file, output_file):
            successful += 1
        else:
            failed += 1

    print("-" * 50)
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} images")
    if failed > 0:
        print(f"Failed to process: {failed} images")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and overlay facial and body keypoints using MediaPipe"
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing images"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to the output folder (optional, defaults to input_folder_keypoints)",
        default=None
    )

    args = parser.parse_args()

    process_folder(args.input_folder, args.output)


if __name__ == "__main__":
    main()