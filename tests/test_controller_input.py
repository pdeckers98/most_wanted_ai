"""Unit tests for controller input reading and logging."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.input.config import ControllerConfig
from src.input.controller import ControllerReader
from src.input.input_logger import InputLogger


class TestControllerConfig:
    """Tests for ControllerConfig validation."""

    def test_valid_defaults(self):
        """Test initialization with default values."""
        cfg = ControllerConfig()
        assert cfg.joystick_index == 0
        assert cfg.steer_axis == 0
        assert cfg.throttle_axis == 5
        assert cfg.brake_axis == 4
        assert cfg.trigger_threshold == 0.5
        assert cfg.steer_deadzone == 0.05

    def test_custom_valid_config(self):
        """Test initialization with custom valid values."""
        cfg = ControllerConfig(
            joystick_index=1,
            steer_axis=0,
            throttle_axis=5,
            brake_axis=4,
            trigger_threshold=0.7,
            steer_deadzone=0.1,
        )
        assert cfg.joystick_index == 1
        assert cfg.trigger_threshold == 0.7
        assert cfg.steer_deadzone == 0.1

    def test_invalid_joystick_index_negative(self):
        """Test that negative joystick_index raises ValueError."""
        with pytest.raises(ValueError, match="joystick_index must be non-negative"):
            ControllerConfig(joystick_index=-1)

    def test_invalid_trigger_threshold_below_range(self):
        """Test that trigger_threshold below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="trigger_threshold must be in"):
            ControllerConfig(trigger_threshold=-0.1)

    def test_invalid_trigger_threshold_above_range(self):
        """Test that trigger_threshold above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="trigger_threshold must be in"):
            ControllerConfig(trigger_threshold=1.1)

    def test_invalid_steer_deadzone_negative(self):
        """Test that negative steer_deadzone raises ValueError."""
        with pytest.raises(ValueError, match="steer_deadzone must be in"):
            ControllerConfig(steer_deadzone=-0.01)

    def test_invalid_steer_deadzone_at_boundary(self):
        """Test that steer_deadzone >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="steer_deadzone must be in"):
            ControllerConfig(steer_deadzone=1.0)

    def test_trigger_threshold_boundaries(self):
        """Test trigger_threshold accepts [0.0, 1.0] inclusive."""
        cfg_min = ControllerConfig(trigger_threshold=0.0)
        assert cfg_min.trigger_threshold == 0.0

        cfg_max = ControllerConfig(trigger_threshold=1.0)
        assert cfg_max.trigger_threshold == 1.0

    def test_steer_deadzone_boundaries(self):
        """Test steer_deadzone accepts [0.0, 1.0) inclusive on lower bound."""
        cfg_min = ControllerConfig(steer_deadzone=0.0)
        assert cfg_min.steer_deadzone == 0.0

        cfg_max = ControllerConfig(steer_deadzone=0.99)
        assert cfg_max.steer_deadzone == 0.99


class TestControllerReader:
    """Tests for ControllerReader initialization and input reading."""

    def test_init_success_single_controller(self):
        """Test successful initialization with one controller connected."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Xbox 360 Controller"

            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            cfg = ControllerConfig()
            ControllerReader(cfg)

            mock_pygame.init.assert_called_once()
            mock_pygame.joystick.init.assert_called_once()
            mock_pygame.joystick.Joystick.assert_called_once_with(0)
            mock_joystick.init.assert_called_once()

    def test_init_no_controller_detected(self):
        """Test RuntimeError when no controller is connected."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_pygame.joystick.get_count.return_value = 0

            cfg = ControllerConfig()
            with pytest.raises(RuntimeError, match="No gamepad detected"):
                ControllerReader(cfg)

    def test_init_joystick_index_out_of_range(self):
        """Test RuntimeError when joystick_index exceeds available controllers."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_pygame.joystick.get_count.return_value = 2

            cfg = ControllerConfig(joystick_index=5)
            with pytest.raises(RuntimeError, match="Joystick index 5 out of range"):
                ControllerReader(cfg)

    def test_read_returns_dict_structure(self):
        """Test that read() returns dict with correct keys."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            # Mock axis reads: steer=0.5, throttle=0.6, brake=0.3
            mock_joystick.get_axis.side_effect = [0.5, 0.6, 0.3]

            cfg = ControllerConfig()
            reader = ControllerReader(cfg)
            state = reader.read()

            assert isinstance(state, dict)
            assert "steer" in state
            assert "throttle" in state
            assert "brake" in state

    def test_read_steering_within_deadzone(self):
        """Test that steering values within deadzone are clamped to 0.0."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            # Steer within deadzone (0.05), throttle/brake inactive
            mock_joystick.get_axis.side_effect = [0.02, 0.0, 0.0]

            cfg = ControllerConfig(steer_deadzone=0.05)
            reader = ControllerReader(cfg)
            state = reader.read()

            assert state["steer"] == 0.0

    def test_read_steering_outside_deadzone(self):
        """Test that steering values outside deadzone are rounded to 4 decimals."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            # Steer outside deadzone, throttle/brake inactive
            mock_joystick.get_axis.side_effect = [0.7234567, 0.0, 0.0]

            cfg = ControllerConfig(steer_deadzone=0.05)
            reader = ControllerReader(cfg)
            state = reader.read()

            assert state["steer"] == 0.7235  # Rounded to 4 decimals

    def test_read_steering_negative(self):
        """Test negative steering (left turn)."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            mock_joystick.get_axis.side_effect = [-0.8, 0.0, 0.0]

            cfg = ControllerConfig()
            reader = ControllerReader(cfg)
            state = reader.read()

            assert state["steer"] == -0.8

    def test_read_throttle_above_threshold(self):
        """Test throttle reads as 1 when axis exceeds threshold."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            # Throttle above threshold (0.5)
            mock_joystick.get_axis.side_effect = [0.0, 0.75, 0.0]

            cfg = ControllerConfig(trigger_threshold=0.5)
            reader = ControllerReader(cfg)
            state = reader.read()

            assert state["throttle"] == 1

    def test_read_throttle_below_threshold(self):
        """Test throttle reads as 0 when axis below threshold."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            # Throttle below threshold
            mock_joystick.get_axis.side_effect = [0.0, 0.3, 0.0]

            cfg = ControllerConfig(trigger_threshold=0.5)
            reader = ControllerReader(cfg)
            state = reader.read()

            assert state["throttle"] == 0

    def test_read_brake_above_threshold(self):
        """Test brake reads as 1 when axis exceeds threshold."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            # Brake above threshold
            mock_joystick.get_axis.side_effect = [0.0, 0.0, 0.9]

            cfg = ControllerConfig(trigger_threshold=0.5)
            reader = ControllerReader(cfg)
            state = reader.read()

            assert state["brake"] == 1

    def test_read_brake_below_threshold(self):
        """Test brake reads as 0 when axis below threshold."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            # Brake below threshold
            mock_joystick.get_axis.side_effect = [0.0, 0.0, 0.2]

            cfg = ControllerConfig(trigger_threshold=0.5)
            reader = ControllerReader(cfg)
            state = reader.read()

            assert state["brake"] == 0

    def test_read_pumps_event_queue(self):
        """Test that read() pumps the pygame event queue."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick
            mock_joystick.get_axis.side_effect = [0.0, 0.0, 0.0]

            cfg = ControllerConfig()
            reader = ControllerReader(cfg)
            reader.read()

            mock_pygame.event.pump.assert_called()

    def test_close(self):
        """Test that close() releases joystick and pygame resources."""
        with patch('src.input.controller.pygame') as mock_pygame:
            mock_joystick = MagicMock()
            mock_joystick.get_name.return_value = "Test Controller"
            mock_pygame.joystick.get_count.return_value = 1
            mock_pygame.joystick.Joystick.return_value = mock_joystick

            cfg = ControllerConfig()
            reader = ControllerReader(cfg)
            reader.close()

            mock_joystick.quit.assert_called_once()
            mock_pygame.quit.assert_called_once()


class TestInputLogger:
    """Tests for InputLogger recording and file output."""

    def test_init(self):
        """Test InputLogger initialization."""
        session_dir = Path("/tmp/test_session")
        logger = InputLogger(session_dir)
        assert logger._session_dir == session_dir
        assert logger._log == []

    def test_record_single_state(self):
        """Test recording a single input state."""
        logger = InputLogger(Path("/tmp/test"))
        state = {"steer": 0.5, "throttle": 1, "brake": 0}
        logger.record(state)

        assert len(logger._log) == 1
        assert logger._log[0] == state

    def test_record_multiple_states(self):
        """Test recording multiple input states maintains order."""
        logger = InputLogger(Path("/tmp/test"))
        states = [
            {"steer": 0.0, "throttle": 0, "brake": 0},
            {"steer": 0.5, "throttle": 1, "brake": 0},
            {"steer": -0.5, "throttle": 0, "brake": 1},
        ]

        for state in states:
            logger.record(state)

        assert len(logger._log) == 3
        assert logger._log == states

    def test_save_creates_json_file(self, tmp_path):
        """Test that save() writes inputs.json to session directory."""
        logger = InputLogger(tmp_path)
        state = {"steer": 0.25, "throttle": 1, "brake": 0}
        logger.record(state)
        logger.save()

        output_file = tmp_path / "inputs.json"
        assert output_file.exists()

    def test_save_writes_valid_json(self, tmp_path):
        """Test that save() produces valid JSON."""
        logger = InputLogger(tmp_path)
        states = [
            {"steer": 0.0, "throttle": 0, "brake": 0},
            {"steer": 0.5, "throttle": 1, "brake": 0},
        ]

        for state in states:
            logger.record(state)
        logger.save()

        output_file = tmp_path / "inputs.json"
        with open(output_file, "r") as f:
            loaded = json.load(f)

        assert loaded == states

    def test_save_compact_format(self, tmp_path):
        """Test that save() uses compact separators (no spaces)."""
        logger = InputLogger(tmp_path)
        logger.record({"steer": 0.0, "throttle": 1, "brake": 0})
        logger.save()

        output_file = tmp_path / "inputs.json"
        with open(output_file, "r") as f:
            content = f.read()

        # Compact format has no spaces after separators
        assert ", " not in content  # Would appear with default separators
        assert ": " not in content

    def test_save_empty_log(self, tmp_path):
        """Test that save() handles empty log (no records)."""
        logger = InputLogger(tmp_path)
        logger.save()

        output_file = tmp_path / "inputs.json"
        with open(output_file, "r") as f:
            loaded = json.load(f)

        assert loaded == []

    def test_frame_alignment(self, tmp_path):
        """Test that frame index in inputs.json aligns with frame N."""
        logger = InputLogger(tmp_path)

        # Simulate 30 frames of input
        for frame_idx in range(30):
            state = {
                "steer": 0.1 * frame_idx,
                "throttle": frame_idx % 2,
                "brake": 0,
            }
            logger.record(state)

        logger.save()

        output_file = tmp_path / "inputs.json"
        with open(output_file, "r") as f:
            loaded = json.load(f)

        assert len(loaded) == 30
        # Verify frame alignment: inputs[N].steer should match expected value for frame N
        for frame_idx in range(30):
            assert loaded[frame_idx]["steer"] == 0.1 * frame_idx
            assert loaded[frame_idx]["throttle"] == frame_idx % 2
