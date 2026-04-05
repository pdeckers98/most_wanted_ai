"""Unit tests for window detection utilities."""

import pytest
from unittest.mock import patch, MagicMock
from src.utils.window import (
    find_window_by_title,
    get_client_region,
    validate_client_size,
)


@patch('src.utils.window.win32gui')
def test_find_window_by_title_success(mock_win32gui):
    """Test finding a window by title substring."""
    mock_win32gui.IsWindowVisible.return_value = True
    mock_win32gui.GetWindowText.side_effect = [
        "Other Window",
        "Need for Speed™ Most Wanted",
        "Another Window",
    ]

    def mock_enum(callback, _):
        callback(0x001, None)
        callback(0x002, None)
        callback(0x003, None)

    mock_win32gui.EnumWindows.side_effect = mock_enum

    hwnd = find_window_by_title("Need for Speed")
    assert hwnd == 0x002


@patch('src.utils.window.win32gui')
def test_find_window_by_title_not_found(mock_win32gui):
    """Test that error is raised when window is not found."""
    mock_win32gui.IsWindowVisible.return_value = True
    mock_win32gui.GetWindowText.return_value = "Some Other Window"

    def mock_enum(callback, _):
        callback(0x001, None)

    mock_win32gui.EnumWindows.side_effect = mock_enum

    with pytest.raises(ValueError, match="No window found"):
        find_window_by_title("NFS Most Wanted")


@patch('src.utils.window.win32gui')
def test_get_client_region(mock_win32gui):
    """Test getting client region in screen coordinates."""
    # GetClientRect returns (0, 0, width, height) for client area
    mock_win32gui.GetClientRect.return_value = (0, 0, 1280, 1024)

    # ClientToScreen converts client coords to screen coords
    mock_win32gui.ClientToScreen.side_effect = [
        (100, 50),   # top-left
        (1380, 1074),  # bottom-right
    ]

    region = get_client_region(0x001)
    assert region == (100, 50, 1380, 1074)


@patch('src.utils.window.win32gui')
def test_validate_client_size_success(mock_win32gui):
    """Test validation when size matches."""
    region = (100, 50, 1380, 1074)
    width, height = validate_client_size(region, 1280, 1024)
    assert width == 1280
    assert height == 1024


def test_validate_client_size_mismatch():
    """Test validation fails when size doesn't match."""
    region = (0, 0, 800, 600)
    with pytest.raises(ValueError, match="Window size mismatch"):
        validate_client_size(region, 1280, 1024)
