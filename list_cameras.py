"""
Utility to detect and list available cameras on the system.
Run this before using multi_camera_capture.py to find camera IDs.
"""

import cv2
import sys


def test_camera(camera_id, max_wait=2):
    """Test if a camera is available and get its properties."""
    try:
        # Try V4L2 backend first (Linux)
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_id)

        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()

            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()

                info = {
                    'id': camera_id,
                    'available': True,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                    'shape': frame.shape if ret else None
                }
                cap.release()
                return info

            cap.release()
    except Exception as e:
        pass

    return None


def list_cameras(max_cameras=10):
    """Scan for available cameras."""
    print("Scanning for cameras...\n")

    available_cameras = []

    for i in range(max_cameras):
        print(f"Testing camera {i}...", end=' ')
        sys.stdout.flush()

        info = test_camera(i)

        if info:
            available_cameras.append(info)
            print(f"✓ FOUND - {info['width']}x{info['height']} @ {info['fps']}fps ({info['backend']})")
        else:
            print("✗ Not available")

    print(f"\n{'='*60}")
    print(f"Found {len(available_cameras)} camera(s)\n")

    if available_cameras:
        print("Camera Configuration for multi_camera_capture.py:")
        print("-" * 60)
        print("camera_configs = [")

        camera_names = ['front_center', 'front_left', 'front_right',
                       'rear_left', 'rear_right', 'rear_center']

        for idx, cam in enumerate(available_cameras):
            name = camera_names[idx] if idx < len(camera_names) else f'camera_{idx}'
            print(f"    {{'id': {cam['id']}, 'name': '{name}', "
                  f"'width': {cam['width']}, 'height': {cam['height']}, "
                  f"'fps': {cam['fps']}}},")

        print("]")
        print("-" * 60)
    else:
        print("No cameras found. Please check:")
        print("  1. Cameras are connected")
        print("  2. Camera permissions (try: sudo chmod 666 /dev/video*)")
        print("  3. Cameras are not in use by another application")
        print("  4. Required drivers are installed")

    return available_cameras


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='List available cameras')
    parser.add_argument('--max', type=int, default=10,
                       help='Maximum camera index to check (default: 10)')
    args = parser.parse_args()

    cameras = list_cameras(max_cameras=args.max)

    print(f"\nTotal cameras found: {len(cameras)}")
