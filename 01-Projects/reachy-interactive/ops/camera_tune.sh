#!/bin/bash
# Camera (Logitech C270) tuning for the hallway demo. V4L2 controls reset when
# the device re-enumerates, so this runs at boot via camera_tune.service.
#
# Why these values (measured on site):
#   exposure_auto_priority=0  the default 1 lets the camera stretch exposure and
#                             drop frame rate in dim light, which smears every
#                             frame while the head is moving. Off = constant
#                             short exposure, far less motion blur.
#   sharpness=140             default 24 is very soft; faces need edges for the
#                             Haar cascade to fire at all.
#   backlight_compensation=1  the corridor has bright glass doors behind people,
#                             which otherwise leaves faces underexposed.
#
# NOTE: the C270 is FIXED FOCUS - there is no focus control in V4L2. If the
# image is soft at conversation distance, the lens bezel has to be turned by
# hand (it rotates) or the lens cleaned.
set -e
DEV=${1:-/dev/video0}

for i in $(seq 1 30); do [ -e "$DEV" ] && break; sleep 1; done

v4l2-ctl -d "$DEV" -c exposure_auto_priority=0 || true
v4l2-ctl -d "$DEV" -c sharpness=140 || true
v4l2-ctl -d "$DEV" -c backlight_compensation=1 || true

echo "camera tuned:"
v4l2-ctl -d "$DEV" -C exposure_auto_priority,sharpness,backlight_compensation,gain 2>/dev/null || true
