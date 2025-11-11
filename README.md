# Multi-Camera Synchronized Capture System

A Python system for capturing synchronized frames from 6 cameras simultaneously, designed for autonomous driving data collection.

## Features

- **Synchronized capture** from multiple cameras with timestamp alignment
- **Thread-based capture** for true parallel frame acquisition
- **Automatic frame dropping** to maintain synchronization
- **Configurable resolution and FPS** per camera
- **Metadata logging** with timestamps and sync information
- **Live preview** of all camera feeds
- **Organized output** with sequence directories and JSON metadata

## Quick Start

### 1. Install Dependencies

```bash
pip install opencv-python numpy
```

For better performance on Linux:
```bash
pip install opencv-contrib-python
```

### 2. Find Available Cameras

```bash
python list_cameras.py
```

This will scan your system and output camera IDs with their capabilities. Example output:
```
Found 6 camera(s)

Camera Configuration for multi_camera_capture.py:
------------------------------------------------------------
camera_configs = [
    {'id': 0, 'name': 'front_center', 'width': 1920, 'height': 1080, 'fps': 30},
    {'id': 1, 'name': 'front_left', 'width': 1920, 'height': 1080, 'fps': 30},
    ...
]
```

### 3. Update Camera Configuration

Edit `multi_camera_capture.py` and update the `camera_configs` list in the `main()` function with your camera IDs.

### 4. Run Capture

```bash
python multi_camera_capture.py
```

Press 'q' to stop recording early, or it will run for the configured duration.

## Usage Examples

### Basic Recording (60 seconds at 10 FPS)

```python
from multi_camera_capture import MultiCameraSystem

camera_configs = [
    {'id': 0, 'name': 'front_center', 'width': 1920, 'height': 1080, 'fps': 30},
    {'id': 2, 'name': 'front_left', 'width': 1920, 'height': 1080, 'fps': 30},
    # ... add more cameras
]

multi_cam = MultiCameraSystem(
    camera_configs=camera_configs,
    output_dir="driving_data",
    max_sync_diff=0.05  # 50ms maximum time difference
)

multi_cam.start()
multi_cam.record(duration=60, target_fps=10, save_images=True)
multi_cam.stop()
```

### Continuous Recording (Until User Stops)

```python
multi_cam.record(duration=None, target_fps=10, save_images=True)
```

### Preview Only (No Saving)

```python
multi_cam.record(duration=30, target_fps=10, save_images=False)
```

### Capture Single Synchronized Frame

```python
frame_data = multi_cam.capture_synchronized_frame()

if frame_data:
    for cam_name, (frame, timestamp) in frame_data['frames'].items():
        print(f"{cam_name}: {frame.shape} at {timestamp}")
```

## Output Structure

```
driving_data/
├── seq_000000/
│   ├── front_center.jpg
│   ├── front_left.jpg
│   ├── front_right.jpg
│   ├── rear_left.jpg
│   ├── rear_right.jpg
│   ├── rear_center.jpg
│   └── metadata.json
├── seq_000001/
│   └── ...
└── ...
```

### Metadata Format

```json
{
  "sequence": 0,
  "sync_time": 1699564231.234567,
  "sync_time_readable": "2024-11-09T15:30:31.234567",
  "max_time_diff_ms": 12.5,
  "cameras": {
    "front_center": {
      "timestamp": 1699564231.230000,
      "offset_ms": -4.5
    },
    "front_left": {
      "timestamp": 1699564231.242500,
      "offset_ms": 8.0
    }
  }
}
```

## Configuration Options

### Camera Configuration

```python
{
    'id': 0,              # Camera device ID (/dev/video0 on Linux)
    'name': 'front_center',  # Descriptive name
    'width': 1920,        # Desired width in pixels
    'height': 1080,       # Desired height in pixels
    'fps': 30             # Camera capture FPS (not save rate)
}
```

### System Configuration

- `output_dir`: Directory to save captured frames
- `max_sync_diff`: Maximum allowed time difference between cameras (seconds)
  - Default: 0.05 (50ms)
  - Increase if you get too many sync warnings
  - Decrease for stricter synchronization

### Recording Options

- `duration`: Recording duration in seconds (None = unlimited)
- `target_fps`: Rate to save frames (not camera FPS)
  - 10 FPS = capture every 100ms
  - 30 FPS = capture every 33ms
- `save_images`: Whether to save images or just show preview

## Synchronization Details

The system ensures synchronization by:

1. **Parallel capture**: Each camera runs in its own thread
2. **Timestamp recording**: Every frame gets a precise timestamp
3. **Sync validation**: Checks that all frames are within `max_sync_diff`
4. **Frame dropping**: Discards old frames to avoid buffer lag
5. **Minimal buffering**: Only keeps 1-2 frames per camera

### Typical Sync Performance

- USB cameras: 10-30ms sync difference
- High-quality industrial cameras: <5ms sync difference
- Webcams: 20-50ms sync difference

## Troubleshooting

### No Cameras Found

```bash
# On Linux, check camera permissions
sudo chmod 666 /dev/video*

# List video devices
ls -la /dev/video*

# Check if camera is recognized
v4l2-ctl --list-devices
```

### Camera Opens But No Frames

- Try different camera IDs (sometimes cameras register multiple times)
- Reduce resolution or FPS
- Close other applications using the cameras
- Try different USB ports (USB 3.0 for high bandwidth)

### Poor Synchronization

- Reduce `target_fps` (less frequent captures = easier to sync)
- Increase `max_sync_diff` threshold
- Use USB 3.0 ports for all cameras
- Use a powered USB hub
- Reduce resolution to lower bandwidth requirements

### Dropped Frames

This is normal and indicates the system is maintaining sync by dropping old buffered frames. However, excessive drops indicate:
- CPU overload: Reduce resolution or FPS
- USB bandwidth issues: Spread cameras across different USB controllers
- Disk I/O bottleneck: Use faster storage (SSD)

### Low FPS Performance

```python
# Option 1: Lower resolution
camera_configs = [
    {'id': 0, 'name': 'front', 'width': 1280, 'height': 720, 'fps': 30},
]

# Option 2: Lower JPEG quality (in save_frame_set method)
cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])  # Default is 95

# Option 3: Save as PNG (lossless, no encoding time)
cv2.imwrite(str(img_path), frame)
```

## Integration with simple_bev

To use captured data with the BEV system:

1. **Calibrate cameras**: Get intrinsic and extrinsic parameters
2. **Load synchronized frames**: Read from sequence directories
3. **Process with vox.py**: Use `unproject_image_to_mem()` to create 3D voxel features

Example integration:
```python
import json
import cv2
from pathlib import Path

# Load sequence
seq_dir = Path("driving_data/seq_000000")
with open(seq_dir / "metadata.json") as f:
    metadata = json.load(f)

# Load frames
frames = {}
for cam_name in metadata['cameras'].keys():
    img_path = seq_dir / f"{cam_name}.jpg"
    frames[cam_name] = cv2.imread(str(img_path))

# Process with BEV pipeline...
```

## Hardware Recommendations

### USB Bandwidth

- 6 cameras at 1920x1080@30fps requires ~1.5 Gbps
- Use USB 3.0 (5 Gbps) or USB 3.1 (10 Gbps)
- Distribute cameras across multiple USB controllers

### Recommended Setup

- **CPU**: 4+ cores for parallel processing
- **RAM**: 8GB+ (frames are buffered in memory)
- **Storage**: SSD for high-speed writes
- **USB Hub**: Powered USB 3.0 hub with per-port power switching

## Performance Tips

1. **Use MJPEG mode**: Already enabled in code (`cv2.VideoWriter_fourcc(*'MJPG')`)
2. **Reduce buffer size**: Already set to 1 frame
3. **Use V4L2 backend on Linux**: Already attempted in code
4. **Process frames in batch**: Save multiple frames before processing metadata
5. **Use separate disk for data**: Avoid I/O contention with OS

## License

MIT License - Feel free to modify for your autonomous driving projects.
