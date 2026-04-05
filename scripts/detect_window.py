#!/usr/bin/env python
"""Utility to enumerate and find windows, particularly for game detection."""

import argparse
import win32gui


def detect_windows(search_term=None):
    """Enumerate all visible windows and print their details.

    Args:
        search_term: Optional substring to filter windows.
    """
    windows = []

    def enum_callback(hwnd, _):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True

            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True

            class_name = win32gui.GetClassName(hwnd)
            rect = win32gui.GetClientRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            if search_term is None or search_term.lower() in title.lower():
                windows.append({
                    'hwnd': hwnd,
                    'title': title,
                    'class': class_name,
                    'width': width,
                    'height': height,
                })
        except Exception:
            pass
        return True

    win32gui.EnumWindows(enum_callback, None)

    if not windows:
        print("No windows found.")
        return

    # Print header
    print(f"{'HWND':<12} {'Class':<25} {'Title':<50} {'Size':<12}")
    print("-" * 99)

    # Print windows
    for w in windows:
        hwnd_str = f"0x{w['hwnd']:08X}"
        size_str = f"{w['width']}x{w['height']}"
        title_repr = repr(w['title'])
        if len(title_repr) > 50:
            title_repr = title_repr[:47] + "..."

        print(f"{hwnd_str:<12} {w['class']:<25} {title_repr:<50} {size_str:<12}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find game windows by title search.'
    )
    parser.add_argument(
        '--search',
        type=str,
        help='Search term to filter windows by title',
    )
    args = parser.parse_args()

    detect_windows(search_term=args.search)
