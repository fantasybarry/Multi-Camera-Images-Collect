"""
Multi-Camera Synchronized Capture System
Captures frames from 6 cameras simultaneously with timestamp synchronization.
Suitable for autonomous driving data collection.
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from pathlib import Path
import queue
import json
from typing import List, Dict, Optional, Tuple


class CameraCapture:
    """Handles individual camera capture in a separate thread."""

    def __init__(self, camera_id: int, name: str, width: int = 1920, height: int = 1080, fps: int = 30):
        self.camera_id = camera_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.last_frame_time = None
        self.frame_count = 0
        self.dropped_frames = 0

    def start(self) -> bool:
        """Initialize and start camera capture."""
        try:
            # Try different backends for better performance
            # CAP_V4L2 for Linux, CAP_DSHOW for Windows, CAP_AVFOUNDATION for macOS
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                print(f"[{self.name}] Trying default backend...")
                self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                print(f"[{self.name}] ERROR: Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Enable hardware acceleration if available
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # Reduce buffer size for lower latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"[{self.name}] Initialized: {actual_width}x{actual_height} @ {actual_fps}fps")

            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

            return True

        except Exception as e:
            print(f"[{self.name}] ERROR during initialization: {e}")
            return False

    def _capture_loop(self):
        """Continuous capture loop running in separate thread."""
        while self.running:
            ret, frame = self.cap.read()

            if ret:
                timestamp = time.time()
                self.frame_count += 1
                self.last_frame_time = timestamp

                # Try to put frame in queue, drop if full (avoid blocking)
                try:
                    self.frame_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    self.dropped_frames += 1
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame, timestamp))
                    except:
                        pass
            else:
                print(f"[{self.name}] WARNING: Failed to read frame")
                time.sleep(0.01)

    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Get latest frame from queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop capture and release resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print(f"[{self.name}] Stopped. Captured: {self.frame_count}, Dropped: {self.dropped_frames}")


class MultiCameraSystem:
    """Manages synchronized capture from multiple cameras."""

    def __init__(self,
                 camera_configs: List[Dict],
                 output_dir: str = "camera_captures",
                 max_sync_diff: float = 0.05):
        """
        Initialize multi-camera system.

        Args:
            camera_configs: List of dicts with keys: 'id', 'name', 'width', 'height', 'fps'
            output_dir: Directory to save captured frames
            max_sync_diff: Maximum allowed time difference between cameras (seconds)
        """
        self.cameras: List[CameraCapture] = []
        self.output_dir = Path(output_dir)
        self.max_sync_diff = max_sync_diff
        self.recording = False
        self.sequence_count = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cameras
        for config in camera_configs:
            camera = CameraCapture(
                camera_id=config['id'],
                name=config['name'],
                width=config.get('width', 1920),
                height=config.get('height', 1080),
                fps=config.get('fps', 30)
            )
            self.cameras.append(camera)

    def start(self) -> bool:
        """Start all cameras."""
        print("Starting cameras...")
        success_count = 0

        for camera in self.cameras:
            if camera.start():
                success_count += 1
            else:
                print(f"Failed to start {camera.name}")

        if success_count == 0:
            print("ERROR: No cameras started successfully!")
            return False

        print(f"Successfully started {success_count}/{len(self.cameras)} cameras")

        # Wait for cameras to warm up
        time.sleep(2.0)

        return True

    def capture_synchronized_frame(self) -> Optional[Dict]:
        """
        Capture frames from all cameras with synchronization.

        Returns:
            Dict with camera names as keys and (frame, timestamp) tuples as values,
            or None if synchronization failed.
        """
        frames_dict = {}
        timestamps = []

        # Collect frames from all cameras
        for camera in self.cameras:
            frame_data = camera.get_frame(timeout=0.2)
            if frame_data is not None:
                frame, timestamp = frame_data
                frames_dict[camera.name] = (frame, timestamp)
                timestamps.append(timestamp)

        # Check if we got frames from all cameras
        if len(frames_dict) != len(self.cameras):
            missing = [cam.name for cam in self.cameras if cam.name not in frames_dict]
            print(f"WARNING: Missing frames from: {missing}")
            return None

        # Check synchronization
        if timestamps:
            time_diff = max(timestamps) - min(timestamps)
            if time_diff > self.max_sync_diff:
                print(f"WARNING: Time difference {time_diff*1000:.2f}ms exceeds threshold")
                # Optionally return None if sync is critical
                # return None

        # Add metadata
        result = {
            'frames': frames_dict,
            'sync_time': np.mean(timestamps),
            'max_time_diff': max(timestamps) - min(timestamps) if timestamps else 0,
            'sequence': self.sequence_count
        }

        self.sequence_count += 1
        return result

    def save_frame_set(self, frame_data: Dict, save_images: bool = True):
        """Save a synchronized frame set to disk."""
        seq_num = frame_data['sequence']
        sync_time = frame_data['sync_time']

        # Create sequence directory
        seq_dir = self.output_dir / f"seq_{seq_num:06d}"
        seq_dir.mkdir(exist_ok=True)

        # Save frames
        if save_images:
            for cam_name, (frame, timestamp) in frame_data['frames'].items():
                img_path = seq_dir / f"{cam_name}.jpg"
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save metadata
        metadata = {
            'sequence': seq_num,
            'sync_time': sync_time,
            'sync_time_readable': datetime.fromtimestamp(sync_time).isoformat(),
            'max_time_diff_ms': frame_data['max_time_diff'] * 1000,
            'cameras': {}
        }

        for cam_name, (_, timestamp) in frame_data['frames'].items():
            metadata['cameras'][cam_name] = {
                'timestamp': timestamp,
                'offset_ms': (timestamp - sync_time) * 1000
            }

        with open(seq_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def record(self, duration: float = None, target_fps: float = 10, save_images: bool = True):
        """
        Record synchronized frames.

        Args:
            duration: Recording duration in seconds (None for unlimited)
            target_fps: Target capture rate (not camera FPS, but save rate)
            save_images: Whether to save images or just show preview
        """
        print(f"\nStarting recording...")
        print(f"Target FPS: {target_fps}, Save images: {save_images}")
        print("Press 'q' to stop recording\n")

        self.recording = True
        start_time = time.time()
        frame_interval = 1.0 / target_fps
        next_capture_time = start_time

        total_frames = 0
        failed_syncs = 0

        try:
            while self.recording:
                current_time = time.time()

                # Check duration limit
                if duration and (current_time - start_time) >= duration:
                    break

                # Capture at target FPS
                if current_time >= next_capture_time:
                    frame_data = self.capture_synchronized_frame()

                    if frame_data:
                        total_frames += 1

                        # Save frames
                        if save_images:
                            self.save_frame_set(frame_data, save_images=True)

                        # Display preview (optional - comment out if running headless)
                        self._display_preview(frame_data)

                        # Print status
                        elapsed = current_time - start_time
                        actual_fps = total_frames / elapsed if elapsed > 0 else 0
                        print(f"\rFrames: {total_frames}, FPS: {actual_fps:.2f}, "
                              f"Sync diff: {frame_data['max_time_diff']*1000:.2f}ms", end='')
                    else:
                        failed_syncs += 1

                    next_capture_time += frame_interval

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Small sleep to avoid busy waiting
                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nRecording interrupted by user")
        finally:
            self.recording = False
            cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            print(f"\n\nRecording complete:")
            print(f"  Duration: {elapsed:.2f}s")
            print(f"  Total frames: {total_frames}")
            print(f"  Failed syncs: {failed_syncs}")
            print(f"  Average FPS: {total_frames/elapsed:.2f}")
            print(f"  Output directory: {self.output_dir}")

    def _display_preview(self, frame_data: Dict):
        """Display preview of all cameras (optional)."""
        frames = []
        for camera in self.cameras:
            if camera.name in frame_data['frames']:
                frame, _ = frame_data['frames'][camera.name]
                # Resize for display
                frame_small = cv2.resize(frame, (640, 360))
                # Add camera name
                cv2.putText(frame_small, camera.name, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(frame_small)

        if frames:
            # Arrange in 2x3 grid
            if len(frames) >= 6:
                row1 = np.hstack(frames[0:3])
                row2 = np.hstack(frames[3:6])
                preview = np.vstack([row1, row2])
            elif len(frames) >= 3:
                row1 = np.hstack(frames[0:3])
                row2 = np.hstack(frames[3:] + [np.zeros_like(frames[0])] * (3 - len(frames[3:])))
                preview = np.vstack([row1, row2])
            else:
                preview = np.hstack(frames)

            cv2.imshow('Multi-Camera Preview', preview)

    def stop(self):
        """Stop all cameras and clean up."""
        print("\nStopping cameras...")
        self.recording = False

        for camera in self.cameras:
            camera.stop()

        cv2.destroyAllWindows()
        print("All cameras stopped.")


def main():
    """Example usage for 6-camera setup."""

    # Define camera configuration
    # Adjust camera IDs based on your system (use `ls /dev/video*` on Linux)
    camera_configs = [
        {'id': 0, 'name': 'front_center', 'width': 1920, 'height': 1080, 'fps': 30},
        {'id': 1, 'name': 'front_left', 'width': 1920, 'height': 1080, 'fps': 30},
        {'id': 2, 'name': 'front_right', 'width': 1920, 'height': 1080, 'fps': 30},
        {'id': 3, 'name': 'rear_left', 'width': 1920, 'height': 1080, 'fps': 30},
        {'id': 4, 'name': 'rear_right', 'width': 1920, 'height': 1080, 'fps': 30},
        {'id': 5, 'name': 'rear_center', 'width': 1920, 'height': 1080, 'fps': 30},
    ]

    # Initialize multi-camera system
    multi_cam = MultiCameraSystem(
        camera_configs=camera_configs,
        output_dir="driving_data",
        max_sync_diff=0.05  # 50ms max difference between cameras
    )

    # Start cameras
    if not multi_cam.start():
        print("Failed to start camera system")
        return

    try:
        # Record for 60 seconds at 10 FPS (captures every 100ms)
        # Set duration=None for unlimited recording
        multi_cam.record(
            duration=60,  # seconds (None for unlimited)
            target_fps=10,  # save rate, not camera FPS
            save_images=True  # set False for preview only
        )
    finally:
        # Always clean up
        multi_cam.stop()


if __name__ == "__main__":
    main()
