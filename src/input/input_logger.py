"""Per-frame input accumulator that persists to JSON at session end."""

import json
from pathlib import Path


class InputLogger:
    """Accumulates per-frame gamepad snapshots and writes a single JSON file.

    Each call to record() appends one entry; the list index equals the frame
    index written by FrameWriter, so frame N and inputs[N] are always aligned.
    """

    def __init__(self, session_dir):
        """Initialise the logger for a recording session.

        Args:
            session_dir: Directory where inputs.json will be written.
        """
        self._session_dir = Path(session_dir)
        self._log = []

    def record(self, state):
        """Append the input state for the current frame.

        Args:
            state: dict returned by ControllerReader.read().
        """
        self._log.append(state)

    def save(self):
        """Write the accumulated log to inputs.json in the session directory."""
        out_path = self._session_dir / "inputs.json"
        with open(out_path, "w") as f:
            json.dump(self._log, f, separators=(",", ":"))
