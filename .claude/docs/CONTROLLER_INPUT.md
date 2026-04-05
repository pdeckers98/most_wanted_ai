# Controller Input Logging

Documentation for capturing and logging player input during gameplay.

## Overview

**Status: Planned for future implementation**

Input capture will be added as a complementary system to the screen capture pipeline. It will be designed to record keyboard and controller inputs synchronized with captured frames.

**Design approach (future):**
- Capture will run alongside the screen capture system in a separate thread
- Input events will be timestamped and logged to disk per-session
- Synchronization via frame timestamps stored in `session_meta.json` (see SCREEN_CAPTURE.md)
- Output format: JSON log with input events and precise frame indices
- Will support both keyboard (e.g., arrow keys, throttle controls) and controller input (via `pygame` or `pynput`)

For now, the screen capture system captures visual data only. Input logging implementation will follow once baseline visual data collection is validated.

## Setup

[Add: Controller configuration, required libraries]

## Input Mapping

[Add: Action space definition, button/axis mapping]

## Recording

[Add: How to run input logger, output format, timestamps]
