"""
Analyze synchronization quality of captured multi-camera data.
Reads metadata files and generates sync statistics and visualizations.
"""

import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from typing import List, Dict


def load_sequence_metadata(data_dir: Path) -> List[Dict]:
    """Load all sequence metadata files from a directory."""
    metadata_list = []

    seq_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('seq_')])

    for seq_dir in seq_dirs:
        metadata_path = seq_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                metadata_list.append(metadata)

    return metadata_list


def analyze_synchronization(metadata_list: List[Dict]) -> Dict:
    """Analyze synchronization quality across all sequences."""
    if not metadata_list:
        return {}

    time_diffs = []
    camera_offsets = {cam: [] for cam in metadata_list[0]['cameras'].keys()}
    frame_times = []

    for metadata in metadata_list:
        time_diffs.append(metadata['max_time_diff_ms'])
        frame_times.append(metadata['sync_time'])

        for cam_name, cam_data in metadata['cameras'].items():
            camera_offsets[cam_name].append(cam_data['offset_ms'])

    # Calculate statistics
    time_diffs = np.array(time_diffs)
    frame_times = np.array(frame_times)

    # Calculate frame rate
    if len(frame_times) > 1:
        time_intervals = np.diff(frame_times)
        actual_fps = 1.0 / np.mean(time_intervals) if len(time_intervals) > 0 else 0
        fps_std = np.std(1.0 / time_intervals) if len(time_intervals) > 0 else 0
    else:
        actual_fps = 0
        fps_std = 0

    stats = {
        'total_sequences': len(metadata_list),
        'duration_seconds': frame_times[-1] - frame_times[0] if len(frame_times) > 1 else 0,
        'actual_fps': actual_fps,
        'fps_std': fps_std,
        'sync_quality': {
            'mean_diff_ms': np.mean(time_diffs),
            'std_diff_ms': np.std(time_diffs),
            'max_diff_ms': np.max(time_diffs),
            'min_diff_ms': np.min(time_diffs),
            'median_diff_ms': np.median(time_diffs),
            'p95_diff_ms': np.percentile(time_diffs, 95),
            'p99_diff_ms': np.percentile(time_diffs, 99),
        },
        'camera_offsets': {}
    }

    # Per-camera statistics
    for cam_name, offsets in camera_offsets.items():
        offsets = np.array(offsets)
        stats['camera_offsets'][cam_name] = {
            'mean_offset_ms': np.mean(offsets),
            'std_offset_ms': np.std(offsets),
            'max_offset_ms': np.max(offsets),
            'min_offset_ms': np.min(offsets),
        }

    return stats


def print_analysis(stats: Dict):
    """Pretty print analysis results."""
    if not stats:
        print("No data to analyze!")
        return

    print("\n" + "="*70)
    print("MULTI-CAMERA SYNCHRONIZATION ANALYSIS")
    print("="*70)

    print(f"\n📊 RECORDING STATISTICS")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"  Actual FPS: {stats['actual_fps']:.2f} ± {stats['fps_std']:.2f}")

    print(f"\n⏱️  SYNCHRONIZATION QUALITY")
    sync = stats['sync_quality']
    print(f"  Mean time difference: {sync['mean_diff_ms']:.2f} ms")
    print(f"  Std deviation: {sync['std_diff_ms']:.2f} ms")
    print(f"  Median: {sync['median_diff_ms']:.2f} ms")
    print(f"  95th percentile: {sync['p95_diff_ms']:.2f} ms")
    print(f"  99th percentile: {sync['p99_diff_ms']:.2f} ms")
    print(f"  Max time difference: {sync['max_diff_ms']:.2f} ms")
    print(f"  Min time difference: {sync['min_diff_ms']:.2f} ms")

    # Quality assessment
    mean_diff = sync['mean_diff_ms']
    if mean_diff < 10:
        quality = "EXCELLENT ✓✓✓"
    elif mean_diff < 20:
        quality = "GOOD ✓✓"
    elif mean_diff < 50:
        quality = "ACCEPTABLE ✓"
    else:
        quality = "POOR ✗"

    print(f"\n  Overall Quality: {quality}")

    print(f"\n📷 PER-CAMERA TIMING OFFSETS")
    for cam_name, cam_stats in stats['camera_offsets'].items():
        print(f"\n  {cam_name}:")
        print(f"    Mean offset: {cam_stats['mean_offset_ms']:+.2f} ms")
        print(f"    Std deviation: {cam_stats['std_offset_ms']:.2f} ms")
        print(f"    Range: [{cam_stats['min_offset_ms']:+.2f}, {cam_stats['max_offset_ms']:+.2f}] ms")

        # Identify consistently early/late cameras
        if cam_stats['mean_offset_ms'] < -5:
            print(f"    ⚠️  Camera tends to capture EARLY")
        elif cam_stats['mean_offset_ms'] > 5:
            print(f"    ⚠️  Camera tends to capture LATE")

    print("\n" + "="*70)

    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    if sync['mean_diff_ms'] > 50:
        print("  • Synchronization is poor. Consider:")
        print("    - Reducing target_fps to give more time between captures")
        print("    - Using USB 3.0 ports for all cameras")
        print("    - Distributing cameras across different USB controllers")
        print("    - Reducing camera resolution or FPS")
    elif sync['p99_diff_ms'] > 100:
        print("  • Occasional sync spikes detected. Consider:")
        print("    - Checking for system load/CPU throttling")
        print("    - Using an SSD for faster I/O")
        print("    - Closing background applications")
    else:
        print("  • Synchronization quality is good!")
        print("  • Data is suitable for multi-camera processing")

    if stats['fps_std'] > 1.0:
        print("  • High FPS variance detected:")
        print("    - Check CPU load during recording")
        print("    - Consider lowering target_fps for more consistent capture")

    print()


def generate_visualization(metadata_list: List[Dict], output_path: str = None):
    """Generate visualization plots (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")
        return

    if not metadata_list:
        print("No data to visualize!")
        return

    # Extract data
    timestamps = [datetime.fromtimestamp(m['sync_time']) for m in metadata_list]
    time_diffs = [m['max_time_diff_ms'] for m in metadata_list]

    camera_names = list(metadata_list[0]['cameras'].keys())
    camera_offsets = {cam: [] for cam in camera_names}

    for metadata in metadata_list:
        for cam_name in camera_names:
            camera_offsets[cam_name].append(metadata['cameras'][cam_name]['offset_ms'])

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Time differences over time
    ax1 = axes[0]
    ax1.plot(timestamps, time_diffs, 'b-', alpha=0.6, linewidth=1)
    ax1.fill_between(timestamps, 0, time_diffs, alpha=0.3)
    ax1.axhline(y=np.mean(time_diffs), color='r', linestyle='--',
                label=f'Mean: {np.mean(time_diffs):.2f}ms')
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5,
                label='50ms threshold')
    ax1.set_ylabel('Max Time Difference (ms)', fontsize=10)
    ax1.set_title('Camera Synchronization Quality Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # Plot 2: Per-camera offsets
    ax2 = axes[1]
    for cam_name, offsets in camera_offsets.items():
        ax2.plot(timestamps, offsets, label=cam_name, alpha=0.7, linewidth=1)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Camera Offset (ms)', fontsize=10)
    ax2.set_title('Individual Camera Timing Offsets', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze synchronization quality of multi-camera capture data'
    )
    parser.add_argument('data_dir', type=str,
                       help='Directory containing captured sequences')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--output', type=str, default='sync_analysis.png',
                       help='Output path for visualization plot')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Directory '{data_dir}' does not exist!")
        return

    print(f"Loading data from: {data_dir}")
    metadata_list = load_sequence_metadata(data_dir)

    if not metadata_list:
        print(f"No sequence data found in {data_dir}")
        return

    print(f"Loaded {len(metadata_list)} sequences")

    # Analyze
    stats = analyze_synchronization(metadata_list)

    # Print results
    print_analysis(stats)

    # Generate plots if requested
    if args.plot:
        generate_visualization(metadata_list, args.output)


if __name__ == "__main__":
    main()
