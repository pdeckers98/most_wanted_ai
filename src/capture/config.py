"""Configuration for the screen capture system."""

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class CaptureConfig:
    """Configuration settings for screen capture.

    Attributes:
        target_fps: Frames per second to capture (default 30).
        source_width: Game window width in pixels (default 1280).
        source_height: Game window height in pixels (default 1024).
        output_width: Downscaled output width (default 480).
        output_height: Downscaled output height (default 384).
        output_dir: Directory to save captured frames (default data/captures).
        hotkey: Keyboard hotkey to toggle recording (default F9).
        window_title_substring: Substring to search for game window title.
        beep_on_start: Enable beep sound when recording starts.
        beep_on_stop: Enable beep sound when recording stops.
    """

    target_fps: int = 30
    source_width: int = 1280
    source_height: int = 1024
    output_width: int = 480
    output_height: int = 384
    output_dir: Path = dataclasses.field(default_factory=lambda: Path("data/captures"))
    hotkey: str = "F9"
    window_title_substring: str = "Need for Speed"
    beep_on_start: bool = True
    beep_on_stop: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {self.target_fps}")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError(
                f"source dimensions must be positive, "
                f"got {self.source_width}x{self.source_height}"
            )
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError(
                f"output dimensions must be positive, "
                f"got {self.output_width}x{self.output_height}"
            )
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
