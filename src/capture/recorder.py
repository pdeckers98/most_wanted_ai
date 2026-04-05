"""Main orchestrator for screen capture - entry point for recording."""

import argparse
import signal
import sys
import threading
import time
import winsound
from datetime import datetime
from pathlib import Path

import keyboard

from src.capture.config import CaptureConfig
from src.capture.grabber import FrameGrabber
from src.capture.preprocessor import preprocess_frame
from src.capture.writer import FrameWriter
from src.utils.window import (
    find_window_by_title,
    get_client_region,
    set_dpi_aware,
    validate_client_size,
)


class ScreenCapture:
    """Main screen capture orchestrator."""

    def __init__(self, config):
        """Initialize screen capture system.

        Args:
            config: CaptureConfig instance.
        """
        self.config = config
        self._recording = threading.Event()
        self._shutdown = threading.Event()
        self._frame_count = 0
        self._session_start = None

        # Find game window
        print(f"Looking for window containing '{config.window_title_substring}'...")
        self._hwnd = find_window_by_title(config.window_title_substring)
        print(f"Found window: HWND=0x{self._hwnd:08X}")

        # Get client region
        region = get_client_region(self._hwnd)
        validate_client_size(region, config.source_width, config.source_height)
        print(f"Client region: {region}")
        self._region = region

        # Initialize components
        self._grabber = None
        self._writer = None

    def _hotkey_toggle(self):
        """Toggle recording state via hotkey."""
        if self._recording.is_set():
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start a new recording session."""
        self._recording.set()
        self._frame_count = 0
        self._session_start = datetime.now()

        # Create session directory
        session_name = self._session_start.strftime("session_%Y%m%d_%H%M%S")
        session_dir = self.config.output_dir / session_name

        # Initialize components
        self._grabber = FrameGrabber(self._region, self.config.target_fps)
        self._writer = FrameWriter(session_dir)

        self._grabber.start()

        # Audio feedback
        if self.config.beep_on_start:
            winsound.Beep(1000, 100)

        print(f"Recording started. Session: {session_name}")
        print(f"Hotkey: {self.config.hotkey} to toggle, Ctrl+C to exit")

    def _stop_recording(self):
        """Stop the current recording session."""
        self._recording.clear()

        if self._grabber:
            self._grabber.stop()
        if self._writer:
            self._writer.flush()
            self._writer.close()

        elapsed = (datetime.now() - self._session_start).total_seconds()

        # Audio feedback
        if self.config.beep_on_stop:
            winsound.Beep(500, 100)

        print(
            f"Recording stopped. "
            f"Frames: {self._frame_count}, "
            f"Duration: {elapsed:.1f}s"
        )

    def _capture_loop(self):
        """Main capture loop - runs while recording."""
        last_status_time = time.time()

        while self._recording.is_set() and not self._shutdown.is_set():
            frame = self._grabber.get_frame()

            if frame is None:
                # No new frame available, skip this iteration
                time.sleep(0.001)
                continue

            # Preprocess: resize and greyscale squeeze
            processed = preprocess_frame(frame)

            # Write to disk (non-blocking queue)
            try:
                self._writer.write(processed, self._frame_count)
                self._frame_count += 1
            except Exception as e:
                print(f"Error writing frame: {e}")

            # Periodic status update
            now = time.time()
            if now - last_status_time > 5.0:
                elapsed = now - time.time()
                fps = self._frame_count / elapsed if elapsed > 0 else 0
                print(f"Frames: {self._frame_count} ({fps:.1f} fps)")
                last_status_time = now

    def run(self):
        """Run the screen capture system."""
        set_dpi_aware()

        # Register hotkey
        keyboard.add_hotkey(self.config.hotkey, self._hotkey_toggle)

        # Set up signal handlers
        def signal_handler(sig, frame):
            self._shutdown.set()
            self._recording.clear()

        signal.signal(signal.SIGINT, signal_handler)

        print(f"Screen Capture Ready")
        print(f"Hotkey: {self.config.hotkey} to start/stop recording")
        print(f"Resolution: {self.config.source_width}x{self.config.source_height}")
        print(f" → {self.config.output_width}x{self.config.output_height}")
        print(f"Target FPS: {self.config.target_fps}")
        print(f"Output dir: {self.config.output_dir}")

        # Main loop
        while not self._shutdown.is_set():
            if self._recording.is_set():
                self._capture_loop()
            else:
                time.sleep(0.1)

        # Cleanup
        if self._recording.is_set():
            self._stop_recording()

        if self._grabber:
            self._grabber.close()

        keyboard.remove_all_hotkeys()
        print("Shutdown complete.")


def main():
    """Entry point for screen capture."""
    parser = argparse.ArgumentParser(
        description="Screen capture for NFS Most Wanted 2005 training data"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second (default 30)",
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        default="F9",
        help="Hotkey to toggle recording (default F9)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/captures"),
        help="Output directory for captures (default data/captures)",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="Need for Speed",
        help="Substring to search for in window title",
    )

    args = parser.parse_args()

    config = CaptureConfig(
        target_fps=args.fps,
        hotkey=args.hotkey,
        output_dir=args.output_dir,
        window_title_substring=args.window_title,
    )

    try:
        capture = ScreenCapture(config)
        capture.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
