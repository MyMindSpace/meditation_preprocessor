import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    raise RuntimeError("OpenCV is required. Install with: pip install opencv-python numpy")


def extract_frames(video_path: Path, target_fps: float = None, max_frames: int = None) -> List[np.ndarray]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        target_fps: Target FPS for frame sampling (None = use original FPS)
        max_frames: Maximum number of frames to extract (None = extract all)
    
    Returns:
        List of frames as numpy arrays (H, W, C)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame sampling interval
    if target_fps and target_fps < original_fps:
        frame_interval = int(original_fps / target_fps)
    else:
        frame_interval = 1
    
    frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample frames based on interval
        if frame_count % frame_interval == 0:
            frames.append(frame.copy())
            extracted_count += 1
            
            # Stop if we've reached max_frames
            if max_frames and extracted_count >= max_frames:
                break
                
        frame_count += 1
    
    cap.release()
    return frames


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Resize frame to target dimensions.
    
    Args:
        frame: Input frame (H, W, C)
        target_size: Target (width, height)
    
    Returns:
        Resized frame
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range.
    
    Args:
        frame: Input frame (H, W, C) with values in [0, 255]
    
    Returns:
        Normalized frame with values in [0, 1]
    """
    return frame.astype(np.float32) / 255.0


def process_video(
    video_path: Path,
    output_dir: Path,
    target_size: Tuple[int, int] = (224, 224),
    target_fps: float = None,
    max_frames: int = None,
    save_individual: bool = False,
) -> None:
    """
    Process a single video file.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save processed frames
        target_size: Target frame dimensions (width, height)
        target_fps: Target FPS for frame sampling
        max_frames: Maximum frames to extract
        save_individual: Whether to save individual frame files
    """
    print(f"Processing: {video_path.name}")
    
    # Extract frames
    frames = extract_frames(video_path, target_fps=target_fps, max_frames=max_frames)
    
    if not frames:
        print(f"  No frames extracted from {video_path.name}")
        return
    
    # Process frames
    processed_frames = []
    for i, frame in enumerate(frames):
        # Resize
        resized = resize_frame(frame, target_size)
        
        # Normalize
        normalized = normalize_frame(resized)
        
        processed_frames.append(normalized)
        
        # Save individual frame if requested
        if save_individual:
            frame_path = output_dir / f"{video_path.stem}_frame_{i:04d}.npy"
            np.save(frame_path, normalized)
    
    # Save all frames as single array
    frames_array = np.stack(processed_frames, axis=0)  # Shape: (T, H, W, C)
    output_path = output_dir / f"{video_path.stem}_frames.npy"
    np.save(output_path, frames_array)
    
    print(f"  Extracted {len(processed_frames)} frames, saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Video Preprocessor (VP) for frame extraction, resizing, normalization")
    parser.add_argument("--input", type=str, default=".", help="Input folder containing video files (default: current directory)")
    parser.add_argument("--output", type=str, default="video_frames", help="Output folder for processed frames")
    parser.add_argument("--size", type=int, nargs=2, default=[224, 224], help="Target frame size (width height)")
    parser.add_argument("--fps", type=float, help="Target FPS for frame sampling (default: use original FPS)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to extract per video")
    parser.add_argument("--individual", action="store_true", help="Save individual frame files in addition to combined array")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    video_paths = [
        Path(root) / f
        for root, _, files in os.walk(input_dir)
        for f in files
        if Path(f).suffix.lower() in video_extensions
    ]
    
    if not video_paths:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"Found {len(video_paths)} video files. Processing...")
    
    for video_path in video_paths:
        # Create output subdirectory mirroring input structure
        rel_parent = video_path.parent.relative_to(input_dir)
        video_output_dir = output_dir / rel_parent
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_video(
                video_path=video_path,
                output_dir=video_output_dir,
                target_size=tuple(args.size),
                target_fps=args.fps,
                max_frames=args.max_frames,
                save_individual=args.individual,
            )
        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")
    
    print(f"Done. Processed frames saved under {output_dir}")


if __name__ == "__main__":
    main()
