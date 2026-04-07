"""Tests for the inference agent and model.

Hardware dependencies (vgamepad, ControllerReader, FrameGrabber, keyboard)
are fully mocked so the tests run without any physical devices or ViGEm.
"""

import threading
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from src.agent.model import RacingAgent
from src.agent.inference import (
    ACTIVATE_FREQ_HZ,
    ACTION_THRESHOLD,
    BEEP_DURATION_MS,
    DEACTIVATE_FREQ_HZ,
    InferenceAgent,
    RB_BUTTON_INDEX,
)

# ============================================================================
# Patch target constants
# ============================================================================

_TORCH_LOAD = 'src.agent.inference.torch.load'
_RACING_AGENT = 'src.agent.inference.RacingAgent'
_FRAME_GRABBER = 'src.agent.inference.FrameGrabber'
_CTRL_READER = 'src.agent.inference.ControllerReader'
_VG = 'src.agent.inference.vg'
_ADD_HOTKEY = 'src.agent.inference.keyboard.add_hotkey'
_BEEP = 'src.agent.inference._beep_async'


def _make_agent() -> InferenceAgent:
    """Build an InferenceAgent with all hardware dependencies mocked."""
    with patch(_TORCH_LOAD) as mock_load, \
         patch(_RACING_AGENT), \
         patch(_FRAME_GRABBER), \
         patch(_CTRL_READER) as mock_ctrl_cls, \
         patch(_VG), \
         patch(_ADD_HOTKEY):

        mock_load.return_value = {}  # train.py saves state_dict directly
        mock_ctrl_cls.return_value._joystick = MagicMock()

        agent = InferenceAgent(
            checkpoint_path='data/checkpoints/best_model.pt',
            region=(0, 0, 1280, 1024),
            device='cpu',
        )

    return agent


# ============================================================================
# RacingAgent — output shapes
# ============================================================================

class TestRacingAgentOutputShapes:
    """Forward pass produces tensors with the expected shapes."""

    def setup_method(self):
        self.model = RacingAgent(pretrained=False)
        self.model.eval()

    def test_steer_shape_single_sample(self):
        steer, _ = self.model(torch.zeros(1, 4, 384, 480))
        assert steer.shape == (1, 1)

    def test_actions_shape_single_sample(self):
        _, actions = self.model(torch.zeros(1, 4, 384, 480))
        assert actions.shape == (1, 2)

    def test_shapes_with_batch(self):
        steer, actions = self.model(torch.zeros(8, 4, 384, 480))
        assert steer.shape == (8, 1)
        assert actions.shape == (8, 2)


# ============================================================================
# RacingAgent — output values
# ============================================================================

class TestRacingAgentOutputValues:
    """Output values respect expected invariants."""

    def setup_method(self):
        self.model = RacingAgent(pretrained=False)
        self.model.eval()

    def test_actions_bounded_by_sigmoid(self):
        _, actions = self.model(torch.randn(4, 4, 384, 480))
        assert (actions >= 0.0).all()
        assert (actions <= 1.0).all()

    def test_steer_not_clamped_by_model(self):
        # No activation on steer head — large weights push it outside [-1, 1].
        # Clamping is the responsibility of InferenceAgent._predict, not the model.
        with torch.no_grad():
            self.model.steer_head.weight.fill_(100.0)
            self.model.steer_head.bias.fill_(100.0)
            steer, _ = self.model(torch.ones(1, 4, 384, 480))
            assert float(steer[0, 0]) > 1.0


# ============================================================================
# InferenceAgent — toggle
# ============================================================================

class TestToggle:
    """F7 toggle logic and audio feedback."""

    def setup_method(self):
        self.agent = _make_agent()

    def test_initially_inactive(self):
        assert self.agent._active is False

    def test_first_toggle_activates(self):
        with patch(_BEEP):
            self.agent._toggle()
        assert self.agent._active is True

    def test_second_toggle_deactivates(self):
        with patch(_BEEP):
            self.agent._toggle()
            self.agent._toggle()
        assert self.agent._active is False

    def test_activate_beeps_high_frequency(self):
        with patch(_BEEP) as mock_beep:
            self.agent._toggle()
        mock_beep.assert_called_once_with(ACTIVATE_FREQ_HZ, BEEP_DURATION_MS)

    def test_deactivate_beeps_low_frequency(self):
        with patch(_BEEP) as mock_beep:
            self.agent._toggle()   # on
            self.agent._toggle()   # off
        assert mock_beep.call_args_list[-1] == call(
            DEACTIVATE_FREQ_HZ, BEEP_DURATION_MS
        )

    def test_toggle_is_thread_safe(self):
        # 100 concurrent toggles from False must finish as False (even flips).
        with patch(_BEEP):
            threads = [
                threading.Thread(target=self.agent._toggle)
                for _ in range(100)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert self.agent._active is False


# ============================================================================
# InferenceAgent — _predict
# ============================================================================

class TestPredict:
    """Steer clamping and throttle/brake thresholding."""

    def setup_method(self):
        self.agent = _make_agent()
        self.stack = np.zeros((4, 384, 480), dtype=np.uint8)

    def _set_model_output(
        self, steer_val: float, throttle_val: float, brake_val: float
    ) -> None:
        self.agent._model = MagicMock()
        self.agent._model.return_value = (
            torch.tensor([[steer_val]]),
            torch.tensor([[throttle_val, brake_val]]),
        )

    def test_steer_clamped_above_one(self):
        self._set_model_output(3.0, 0.8, 0.1)
        steer, _, _ = self.agent._predict(self.stack)
        assert steer == pytest.approx(1.0)

    def test_steer_clamped_below_minus_one(self):
        self._set_model_output(-5.0, 0.8, 0.1)
        steer, _, _ = self.agent._predict(self.stack)
        assert steer == pytest.approx(-1.0)

    def test_steer_in_range_unchanged(self):
        self._set_model_output(0.42, 0.8, 0.1)
        steer, _, _ = self.agent._predict(self.stack)
        assert steer == pytest.approx(0.42, abs=1e-4)

    def test_throttle_above_threshold_is_one(self):
        self._set_model_output(0.0, ACTION_THRESHOLD + 0.1, 0.1)
        _, throttle, _ = self.agent._predict(self.stack)
        assert throttle == 1

    def test_throttle_below_threshold_is_zero(self):
        self._set_model_output(0.0, ACTION_THRESHOLD - 0.1, 0.1)
        _, throttle, _ = self.agent._predict(self.stack)
        assert throttle == 0

    def test_brake_above_threshold_is_one(self):
        self._set_model_output(0.0, 0.1, ACTION_THRESHOLD + 0.1)
        _, _, brake = self.agent._predict(self.stack)
        assert brake == 1

    def test_brake_below_threshold_is_zero(self):
        self._set_model_output(0.0, 0.8, ACTION_THRESHOLD - 0.1)
        _, _, brake = self.agent._predict(self.stack)
        assert brake == 0


# ============================================================================
# InferenceAgent — _send
# ============================================================================

class TestSend:
    """_send writes correct values to the virtual gamepad."""

    def setup_method(self):
        self.agent = _make_agent()
        self.agent._vpad = MagicMock()

    def test_steering_sent_to_left_joystick(self):
        self.agent._send(0.5, 0, 0)
        self.agent._vpad.left_joystick_float.assert_called_once_with(
            x_value_float=0.5, y_value_float=0.0
        )

    def test_throttle_sent_to_right_trigger(self):
        self.agent._send(0.0, 1, 0)
        self.agent._vpad.right_trigger_float.assert_called_once_with(
            value_float=1.0
        )

    def test_brake_sent_to_left_trigger(self):
        self.agent._send(0.0, 0, 1)
        self.agent._vpad.left_trigger_float.assert_called_once_with(
            value_float=1.0
        )

    def test_update_always_called(self):
        self.agent._send(0.0, 0, 0)
        self.agent._vpad.update.assert_called_once()

    def test_neutral_sends_all_zeros(self):
        self.agent._send(0.0, 0, 0)
        self.agent._vpad.left_joystick_float.assert_called_once_with(
            x_value_float=0.0, y_value_float=0.0
        )
        self.agent._vpad.right_trigger_float.assert_called_once_with(value_float=0.0)
        self.agent._vpad.left_trigger_float.assert_called_once_with(value_float=0.0)


# ============================================================================
# InferenceAgent — _rb_held
# ============================================================================

class TestRbHeld:
    """RB button detection."""

    def setup_method(self):
        self.agent = _make_agent()

    def test_returns_false_when_not_pressed(self):
        self.agent._joystick.get_button.return_value = 0
        assert self.agent._rb_held() is False

    def test_returns_true_when_pressed(self):
        self.agent._joystick.get_button.return_value = 1
        assert self.agent._rb_held() is True

    def test_queries_correct_button_index(self):
        self.agent._joystick.get_button.return_value = 0
        self.agent._rb_held()
        self.agent._joystick.get_button.assert_called_with(RB_BUTTON_INDEX)
