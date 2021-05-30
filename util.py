import cv2
import numpy as np
import torch


def load_image(path):
    img = cv2.imread(path)  # BGR

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = np.stack((binary,) * 3, axis=-1)
    # cv2.imshow("222", img)
    # cv2.waitKey(0)

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[ 0] = x[0] - x[ 2] / 2  # top left x
    y[ 1] = x[1] - x[ 3] / 2  # top left y
    y[ 2] = x[0] + x[ 2] / 2  # bottom right x
    y[ 3] = x[1] + x[ 3] / 2  # bottom right y
    return y