import numpy as np
import cv2


def raw2bgr(data):
    return ycrcb2bgr(raw2ycrcb(data))


def raw2ycrcb(data):
    Y = data[1::2]
    Cb = data[::4]
    Cr = data[2::4]

    Cb = np.repeat(Cb, 2)
    Cr = np.repeat(Cr, 2)

    return np.array((Y, Cr, Cb)).T


def ycrcb2bgr(ycrcb):
    BGR = cv2.cvtColor(ycrcb.reshape((144, 174, 3)), cv2.COLOR_YCrCb2BGR)
    B, G, R = BGR.T

    # Y, Cr, Cb = ycrcb.T
    # Y = Y.astype(np.int)
    # Cr = Cr.astype(np.int)
    # Cb = Cb.astype(np.int)
    #
    # R = Y + 1.402 * (Cr - 128)
    # G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    # B = Y + 1.772 * (Cb - 128)
    #
    # B = B.astype(np.uint8)
    # G = G.astype(np.uint8)
    # R = R.astype(np.uint8)

    B, G, R = R, B, G

    return np.array((B, G, R)).T.reshape((144, 174, 3))
