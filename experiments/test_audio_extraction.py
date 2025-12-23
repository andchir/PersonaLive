#!/usr/bin/env python
"""
Test script to verify audio extraction from driving video works correctly.

This script tests:
1. Saving videos without audio (backward compatibility)
2. Saving videos with audio from a source video
3. Trimming audio when output video is shorter than source
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

try:
    import av
except ImportError:
    print("ERROR: PyAV is required. Install with: pip install av")
    sys.exit(1)


def create_test_video_with_audio(output_path, duration_seconds=5, fps=25, width=256, height=256):
    """Create a test video with audio for testing purposes."""
    container = av.open(output_path, "w")

    # Add video stream
    video_stream = container.add_stream("libx264", rate=fps)
    video_stream.width = width
    video_stream.height = height
    video_stream.pix_fmt = "yuv420p"

    # Add audio stream (AAC, 44100 Hz, stereo)
    audio_stream = container.add_stream("aac", rate=44100)
    audio_stream.layout = "stereo"

    # Generate video frames (simple color gradient)
    num_frames = int(duration_seconds * fps)
    for i in range(num_frames):
        # Create a simple frame with varying color
        r = int(255 * i / num_frames)
        frame_data = np.full((height, width, 3), [r, 128, 255 - r], dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame.pts = i
        for packet in video_stream.encode(frame):
            container.mux(packet)

    # Flush video encoder
    for packet in video_stream.encode():
        container.mux(packet)

    # Generate audio (simple sine wave)
    sample_rate = 44100
    num_samples = int(duration_seconds * sample_rate)
    samples_per_frame = 1024

    t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
    # 440 Hz sine wave (A4 note)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    # Make stereo
    audio_data = np.column_stack([audio_data, audio_data])

    pts = 0
    for i in range(0, num_samples, samples_per_frame):
        chunk = audio_data[i:i + samples_per_frame]
        if len(chunk) == 0:
            break
        frame = av.AudioFrame.from_ndarray(chunk.T, format="s16", layout="stereo")
        frame.sample_rate = sample_rate
        frame.pts = pts
        pts += len(chunk)
        for packet in audio_stream.encode(frame):
            container.mux(packet)

    # Flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)

    container.close()
    print(f"Created test video with audio: {output_path} ({duration_seconds}s)")


def check_video_has_audio(video_path):
    """Check if a video file has an audio track."""
    try:
        container = av.open(video_path)
        audio_streams = [s for s in container.streams if s.type == "audio"]
        has_audio = len(audio_streams) > 0

        if has_audio:
            audio_stream = audio_streams[0]
            duration = float(audio_stream.duration * audio_stream.time_base) if audio_stream.duration else 0
            print(f"  Audio: YES (duration: {duration:.2f}s, sample_rate: {audio_stream.rate})")
        else:
            print(f"  Audio: NO")

        container.close()
        return has_audio
    except Exception as e:
        print(f"  Error checking audio: {e}")
        return False


def get_video_duration(video_path):
    """Get video duration in seconds."""
    try:
        container = av.open(video_path)
        video_stream = next(s for s in container.streams if s.type == "video")
        duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0
        container.close()
        return duration
    except Exception as e:
        print(f"  Error getting duration: {e}")
        return 0


def test_save_videos_from_pil():
    """Test the save_videos_from_pil function with audio."""
    from src.utils.util import save_videos_from_pil

    print("\n" + "=" * 60)
    print("Testing save_videos_from_pil with audio extraction")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images (simulating PIL images from video generation)
        num_frames = 50  # 2 seconds at 25fps
        test_images = []
        for i in range(num_frames):
            r = int(255 * i / num_frames)
            img_data = np.full((256, 256, 3), [255 - r, r, 128], dtype=np.uint8)
            test_images.append(Image.fromarray(img_data))

        # Test 1: Save without audio (backward compatibility)
        print("\nTest 1: Save without audio (backward compatibility)")
        output_path_1 = os.path.join(tmpdir, "output_no_audio.mp4")
        save_videos_from_pil(test_images, output_path_1, fps=25, crf=23)
        print(f"  Output: {output_path_1}")
        has_audio_1 = check_video_has_audio(output_path_1)
        assert not has_audio_1, "Video should not have audio when no source provided"
        print("  PASS: No audio as expected")

        # Create source video with audio (5 seconds)
        source_video_path = os.path.join(tmpdir, "source_with_audio.mp4")
        create_test_video_with_audio(source_video_path, duration_seconds=5)

        # Test 2: Save with audio from source (output shorter than source)
        print("\nTest 2: Save with audio (output 2s, source 5s - should trim audio)")
        output_path_2 = os.path.join(tmpdir, "output_with_audio.mp4")
        save_videos_from_pil(test_images, output_path_2, fps=25, crf=23, audio_source=source_video_path)
        print(f"  Output: {output_path_2}")
        has_audio_2 = check_video_has_audio(output_path_2)
        assert has_audio_2, "Video should have audio when source provided"
        print("  PASS: Has audio as expected")

        # Test 3: Save with longer output (more frames than source duration)
        print("\nTest 3: Save with audio (output 4s, source 5s)")
        long_images = test_images * 2  # 100 frames = 4 seconds at 25fps
        output_path_3 = os.path.join(tmpdir, "output_longer.mp4")
        save_videos_from_pil(long_images, output_path_3, fps=25, crf=23, audio_source=source_video_path)
        print(f"  Output: {output_path_3}")
        has_audio_3 = check_video_has_audio(output_path_3)
        assert has_audio_3, "Video should have audio"
        print("  PASS: Has audio as expected")

        # Test 4: Invalid audio source (should not crash)
        print("\nTest 4: Invalid audio source (should gracefully handle)")
        output_path_4 = os.path.join(tmpdir, "output_invalid_source.mp4")
        save_videos_from_pil(test_images, output_path_4, fps=25, crf=23, audio_source="/nonexistent/video.mp4")
        print(f"  Output: {output_path_4}")
        # Should still create video without audio
        assert os.path.exists(output_path_4), "Video should be created even with invalid source"
        print("  PASS: Gracefully handled invalid source")

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)


if __name__ == "__main__":
    test_save_videos_from_pil()
