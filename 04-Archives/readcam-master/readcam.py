import serial
import time

import numpy as np
import cv2

from cvt import raw2bgr


s = serial.Serial(
    '/dev/tty.usbmodem00000000001A',
    baudrate=1000000
)


buff = b''
chunk_size = 256
delimiter = b'start'

t0 = time.time()

lol = []

try:
    while True:
        data = s.read(chunk_size)
        buff += data

        left = buff.find(delimiter)
        right = buff.rfind(delimiter)
        if left != -1 and right != -1 and left != right:
            img = buff[left + len(delimiter):right]
            img = np.frombuffer(img, dtype=np.uint8)
            # lol.append(img)
            # grey = img[1::2].copy()
            # grey = grey.reshape((144, 174))
            # grey = cv2.resize(grey, (1440, 1740))

            bgr = raw2bgr(img)
            bgr = bgr.reshape((144, 174, 3))

            t = time.time()
            cv2.imshow('video', bgr)
            cv2.waitKey(1)
            # print('Got img', bgr, 1.0 / (t - t0))
            t0 = t

            buff = buff[right:]
except KeyboardInterrupt:
    np.save('ocv.npy', np.array(lol))
