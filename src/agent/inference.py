"""
Real-time inference agent for NFS Most Wanted 2005.

Captures live frames, runs the trained ResNet-18 policy, and sends
actions to a virtual Xbox controller via vgamepad (ViGEm). The physical
controller is always read by this script; the virtual controller is what
the game should be configured to use.

NOTE: On first run, go into the game's controller options and reassign
      controls to the new virtual device that appears in Windows.

Controls:
  F7  - Toggle agent on / off
  RB  - Hold to override the agent with your own input

Usage:
    python src/agent/inference.py
    python src/agent/inference.py --checkpoint data/checkpoints/best_model.pt
    python src/agent/inference.py --left 0 --top 0 --right 1280 --bottom 1024
"""

import argparse
import collections
import logging
import threading
import time

import keyboard
import numpy as np
import torch
import vgamepad as vg
import winsound

from src.agent.model import RacingAgent
from src.capture.grabber import FrameGrabber
from src.capture.preprocessor import preprocess_frame
from src.input.config import ControllerConfig
from src.input.controller import ControllerReader


# ============================================================================
# Constants
# ============================================================================

FRAME_BUFFER_SIZE = 4       # Number of stacked frames the model expects
ACTIVATE_FREQ_HZ = 880      # Beep frequency when agent is turned on
DEACTIVATE_FREQ_HZ = 440    # Beep frequency when agent is turned off
BEEP_DURATION_MS = 200      # Beep length in milliseconds
RB_BUTTON_INDEX = 5         # Right bumper on a standard Xbox controller
ACTION_THRESHOLD = 0.5      # Sigmoid threshold for throttle / brake output


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _beep_async(freq: int, duration_ms: int) -> None:
    """Play a Windows beep in a background thread (non-blocking)."""
    threading.Thread(
        target=winsound.Beep,
        args=(freq, duration_ms),
        daemon=True,
    ).start()


# ============================================================================
# Inference Agent
# ============================================================================

class InferenceAgent:
    """
    Real-time racing agent.

    Reads frames from the screen, runs the policy model, and writes
    actions to a virtual Xbox controller (vgamepad / ViGEm).

    State machine:
      Agent OFF  -> pass physical controller input through to vgamepad
      Agent ON   -> model controls vgamepad
      Agent ON + RB held -> physical input overrides model (pass-through)
    """

    def __init__(
        self,
        checkpoint_path: str,
        region: tuple,
        target_fps: int = 30,
        device: str = 'cuda',
    ):
        self._device = torch.device(device)
        self._frame_interval = 1.0 / target_fps
        self._active = False
        self._lock = threading.Lock()

        # ---- Model --------------------------------------------------------
        logger.info("Loading checkpoint: %s", checkpoint_path)
        self._model = RacingAgent(pretrained=False).to(self._device)
        ckpt = torch.load(checkpoint_path, map_location=self._device)
        self._model.load_state_dict(ckpt['model_state_dict'])
        self._model.eval()
        logger.info("Model loaded and ready.")

        # ---- Frame capture ------------------------------------------------
        self._grabber = FrameGrabber(region=region, target_fps=target_fps)
        self._buffer = collections.deque(maxlen=FRAME_BUFFER_SIZE)

        # ---- Physical controller ------------------------------------------
        ctrl_cfg = ControllerConfig()
        self._controller = ControllerReader(ctrl_cfg)
        # Keep a direct reference for button polling after event pump
        self._joystick = self._controller._joystick

        # ---- Virtual controller output ------------------------------------
        self._vpad = vg.VX360Gamepad()
        logger.info("Virtual gamepad created.")

        # ---- F7 hotkey ----------------------------------------------------
        keyboard.add_hotkey('f7', self._toggle)
        logger.info("Ready. Press F7 to activate the agent.")

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _toggle(self) -> None:
        """Toggle agent on/off (called from keyboard listener thread)."""
        with self._lock:
            self._active = not self._active
            active = self._active

        if active:
            logger.info("Agent ACTIVATED")
            _beep_async(ACTIVATE_FREQ_HZ, BEEP_DURATION_MS)
        else:
            logger.info("Agent DEACTIVATED")
            _beep_async(DEACTIVATE_FREQ_HZ, BEEP_DURATION_MS)

    def _rb_held(self) -> bool:
        """Return True if the right bumper is currently pressed.

        Must be called after self._controller.read() so pygame events
        have already been pumped.
        """
        return bool(self._joystick.get_button(RB_BUTTON_INDEX))

    def _predict(self, frame_stack: np.ndarray) -> tuple:
        """Run one forward pass through the policy network.

        Args:
            frame_stack: (4, 384, 480) uint8 array

        Returns:
            (steer, throttle, brake) — float in [-1,1] and ints 0/1
        """
        tensor = (
            torch.from_numpy(frame_stack).float() / 255.0
        ).unsqueeze(0).to(self._device)   # (1, 4, 384, 480)

        with torch.no_grad():
            steer_t, actions_t = self._model(tensor)

        steer = float(steer_t[0, 0].cpu())
        steer = max(-1.0, min(1.0, steer))  # clamp to valid range

        throttle = int(float(actions_t[0, 0].cpu()) > ACTION_THRESHOLD)
        brake = int(float(actions_t[0, 1].cpu()) > ACTION_THRESHOLD)

        return steer, throttle, brake

    def _send(self, steer: float, throttle: int, brake: int) -> None:
        """Write one action frame to the virtual gamepad."""
        self._vpad.left_joystick_float(
            x_value_float=steer, y_value_float=0.0
        )
        self._vpad.right_trigger_float(value_float=float(throttle))
        self._vpad.left_trigger_float(value_float=float(brake))
        self._vpad.update()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Start the inference loop. Blocks until Ctrl-C."""
        self._grabber.start()
        logger.info("Capture started. Ctrl-C to quit.")

        try:
            while True:
                t0 = time.monotonic()

                # --- Grab frame --------------------------------------------
                frame = self._grabber.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                processed = preprocess_frame(frame)   # (384, 480) uint8

                # Pad buffer so we always have 4 frames from the first step
                if not self._buffer:
                    for _ in range(FRAME_BUFFER_SIZE):
                        self._buffer.append(processed)
                else:
                    self._buffer.append(processed)

                stack = np.stack(self._buffer, axis=0)  # (4, 384, 480)

                # --- Read physical controller (pumps pygame events) --------
                physical = self._controller.read()
                rb = self._rb_held()

                with self._lock:
                    active = self._active

                # --- Decide action -----------------------------------------
                if not active or rb:
                    # Manual: pass physical input straight to vgamepad
                    self._send(
                        physical['steer'],
                        physical['throttle'],
                        physical['brake'],
                    )
                else:
                    # Agent: model drives
                    steer, throttle, brake = self._predict(stack)
                    self._send(steer, throttle, brake)

                # --- Pace the loop to target FPS ---------------------------
                elapsed = time.monotonic() - t0
                wait = self._frame_interval - elapsed
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._send(0.0, 0, 0)   # neutral before exit
            self._grabber.stop()
            self._grabber.close()
            self._controller.close()
            logger.info("Agent stopped.")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run NFS Most Wanted 2005 inference agent'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='data/checkpoints/best_model.pt',
        help='Path to model checkpoint (default: data/checkpoints/best_model.pt)',
    )
    parser.add_argument(
        '--left', type=int, default=0,
        help='Capture region left edge in screen pixels (default: 0)',
    )
    parser.add_argument(
        '--top', type=int, default=0,
        help='Capture region top edge in screen pixels (default: 0)',
    )
    parser.add_argument(
        '--right', type=int, default=1280,
        help='Capture region right edge in screen pixels (default: 1280)',
    )
    parser.add_argument(
        '--bottom', type=int, default=1024,
        help='Capture region bottom edge in screen pixels (default: 1024)',
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Target inference frames per second (default: 30)',
    )
    parser.add_argument(
        '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
        help='PyTorch device (default: cuda)',
    )

    args = parser.parse_args()

    agent = InferenceAgent(
        checkpoint_path=args.checkpoint,
        region=(args.left, args.top, args.right, args.bottom),
        target_fps=args.fps,
        device=args.device,
    )
    agent.run()


if __name__ == '__main__':
    main()
