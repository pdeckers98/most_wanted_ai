"""Window detection and region calculation utilities for screen capture."""

import ctypes
import win32gui


def set_dpi_aware():
    """Enable DPI awareness to get accurate pixel coordinates on Windows 11."""
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def find_window_by_title(substring):
    """Find a window by title substring.

    Args:
        substring: String to search for in window titles (case-insensitive).

    Returns:
        int: Window handle (HWND) of the first matching window.

    Raises:
        ValueError: If no window with matching title is found.
    """
    hwnd = None

    def enum_callback(handle, _):
        nonlocal hwnd
        try:
            title = win32gui.GetWindowText(handle)
            if substring.lower() in title.lower() and win32gui.IsWindowVisible(handle):
                hwnd = handle
                return False
        except Exception:
            pass
        return True

    win32gui.EnumWindows(enum_callback, None)

    if hwnd is None:
        raise ValueError(f"No window found with title containing '{substring}'")

    return hwnd


def get_client_region(hwnd):
    """Get the client area region in screen coordinates.

    Args:
        hwnd: Window handle to query.

    Returns:
        tuple: (left, top, right, bottom) in screen coordinates.
    """
    rect = win32gui.GetClientRect(hwnd)
    left, top = rect[0], rect[1]
    right, bottom = rect[2], rect[3]

    # Convert client coordinates to screen coordinates
    top_left = win32gui.ClientToScreen(hwnd, (left, top))
    bottom_right = win32gui.ClientToScreen(hwnd, (right, bottom))

    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])


def validate_client_size(region, expected_width, expected_height):
    """Validate that the client area matches expected dimensions.

    Args:
        region: (left, top, right, bottom) tuple.
        expected_width: Expected width in pixels.
        expected_height: Expected height in pixels.

    Returns:
        tuple: (actual_width, actual_height)

    Raises:
        ValueError: If dimensions don't match expected values.
    """
    left, top, right, bottom = region
    actual_width = right - left
    actual_height = bottom - top

    if actual_width != expected_width or actual_height != expected_height:
        raise ValueError(
            f"Window size mismatch. Expected {expected_width}x{expected_height}, "
            f"got {actual_width}x{actual_height}"
        )

    return (actual_width, actual_height)
