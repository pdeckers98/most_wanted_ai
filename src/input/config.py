"""Configuration for gamepad input reading."""

import dataclasses


@dataclasses.dataclass
class ControllerConfig:
    """Configuration settings for gamepad input capture.

    Attributes:
        joystick_index: Pygame joystick index (default 0 = first controller).
        steer_axis: Axis index for steering (default 0 = left stick X).
        throttle_axis: Axis index for throttle (default 5 = right trigger RT).
        brake_axis: Axis index for brake (default 4 = left trigger LT).
        trigger_threshold: Axis value above which a trigger counts as pressed (default 0.5).
        steer_deadzone: Steering axis values within this range are clamped to 0.0 (default 0.05).
    """

    joystick_index: int = 0
    steer_axis: int = 0
    throttle_axis: int = 5
    brake_axis: int = 4
    trigger_threshold: float = 0.5
    steer_deadzone: float = 0.05

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.joystick_index < 0:
            raise ValueError(
                f"joystick_index must be non-negative, got {self.joystick_index}"
            )
        if not 0.0 <= self.trigger_threshold <= 1.0:
            raise ValueError(
                f"trigger_threshold must be in [0.0, 1.0], got {self.trigger_threshold}"
            )
        if not 0.0 <= self.steer_deadzone < 1.0:
            raise ValueError(
                f"steer_deadzone must be in [0.0, 1.0), got {self.steer_deadzone}"
            )
