"""Gamepad state reader using pygame."""

import pygame


class ControllerReader:
    """Reads gamepad axis and trigger state via pygame joystick API.

    Steering is returned as a continuous float in [-1.0, 1.0].
    Throttle and brake are returned as discrete ints (0 or 1) based on
    whether the corresponding trigger axis exceeds the configured threshold.
    """

    def __init__(self, config):
        """Initialise pygame and open the configured joystick.

        Args:
            config: ControllerConfig instance.

        Raises:
            RuntimeError: If no controller is connected or the index is out of range.
        """
        self._config = config

        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError(
                "No gamepad detected. Connect a controller and retry."
            )
        if config.joystick_index >= count:
            raise RuntimeError(
                f"Joystick index {config.joystick_index} out of range "
                f"({count} controller(s) found)."
            )

        self._joystick = pygame.joystick.Joystick(config.joystick_index)
        self._joystick.init()
        print(f"Controller: {self._joystick.get_name()} (index {config.joystick_index})")

    def read(self):
        """Sample the current gamepad state.

        Pumps the pygame event queue so joystick readings stay fresh, then
        reads steering, throttle, and brake.

        Returns:
            dict with keys:
                steer (float): Steering in [-1.0, 1.0]; negative = left, positive = right.
                throttle (int): 1 if RT axis > threshold, else 0.
                brake (int): 1 if LT axis > threshold, else 0.
        """
        pygame.event.pump()

        cfg = self._config

        raw_steer = self._joystick.get_axis(cfg.steer_axis)
        steer = 0.0 if abs(raw_steer) < cfg.steer_deadzone else round(raw_steer, 4)

        throttle_raw = self._joystick.get_axis(cfg.throttle_axis)
        throttle = 1 if throttle_raw > cfg.trigger_threshold else 0

        brake_raw = self._joystick.get_axis(cfg.brake_axis)
        brake = 1 if brake_raw > cfg.trigger_threshold else 0

        return {"steer": steer, "throttle": throttle, "brake": brake}

    def close(self):
        """Release joystick and pygame resources."""
        self._joystick.quit()
        pygame.quit()
