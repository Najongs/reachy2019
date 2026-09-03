#!/usr/bin/env python3
"""Turn up the ReSpeaker 4 Mic Array's onboard gain (AGC) for far speech.

The array (USB 2886:0018) has no ALSA volume control - its gain lives in the
XMOS DSP, reachable only through a vendor USB control interface (protocol from
the official respeaker/usb_4_mic_array tuning.py). The DSP resets to defaults
on power-up, so this runs at boot via respeaker_gain.service.

Usage (needs USB access - run with sudo, or via the systemd unit):
    sudo python3 respeaker_gain.py            # show current AGC values
    sudo python3 respeaker_gain.py --apply    # AGC on, max gain 30 dB
"""

import argparse
import struct
import sys
import time

# name: (register id, offset, type)  - subset of the official parameter table
PARAMS = {
    'AGCONOFF':       (19, 0, 'int'),    # 0=fixed gain, 1=auto
    'AGCMAXGAIN':     (19, 1, 'float'),  # linear, 1..1000 (0..60 dB)
    'AGCDESIREDLEVEL': (19, 2, 'float'), # target output level, 0..0.99
    'AGCGAIN':        (19, 3, 'float'),  # current/initial gain, linear
    'AGCTIME':        (19, 4, 'float'),  # ramp time in seconds
}
TIMEOUT = 3000


def find_dev():
    import usb.core
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if dev is None:
        sys.exit('ReSpeaker 4 Mic Array (2886:0018) not found')
    return dev


def read_param(dev, name):
    import usb.util
    reg, offset, kind = PARAMS[name]
    cmd = 0x80 | offset
    if kind == 'int':
        cmd |= 0x40
    data = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, cmd, reg, 8, TIMEOUT)
    lo, hi = struct.unpack('ii', data.tobytes())
    return lo if kind == 'int' else lo * (2.0 ** hi)


def write_param(dev, name, value):
    import usb.util
    reg, offset, kind = PARAMS[name]
    if kind == 'int':
        payload = struct.pack('iii', offset, int(value), 1)
    else:
        payload = struct.pack('ifi', offset, float(value), 0)
    dev.ctrl_transfer(
        usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0, 0, reg, payload, TIMEOUT)


def show(dev):
    for name in PARAMS:
        try:
            print('  {:16} = {:.3f}'.format(name, float(read_param(dev, name))))
        except Exception as e:
            print('  {:16} read failed: {}'.format(name, e))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true',
                        help='enable AGC and raise the gain ceiling to ~30 dB')
    parser.add_argument('--max-gain-db', type=float, default=50.0,
                        help='AGC max gain in dB (default 50, hardware cap 60)')
    parser.add_argument('--desired-level', type=float, default=0.05,
                        help='AGC target output level 0..0.99; raise so soft/'
                             'far voices are captured louder (default 0.05)')
    args = parser.parse_args()

    dev = find_dev()
    print('before:')
    show(dev)

    if args.apply:
        linear = 10.0 ** (args.max_gain_db / 20.0)
        write_param(dev, 'AGCONOFF', 1)
        write_param(dev, 'AGCMAXGAIN', linear)
        write_param(dev, 'AGCDESIREDLEVEL', args.desired_level)
        time.sleep(0.2)
        print('after:')
        show(dev)
        print('AGC on, max gain {:.0f} dB, desired {:.3f}'.format(
            args.max_gain_db, args.desired_level))


if __name__ == '__main__':
    main()
