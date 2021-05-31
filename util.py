import cv2
import numpy as np
from torchvision.transforms import *


def load_image(path):
    img = cv2.imread(path)  # BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img = np.stack((binary,) * 3, axis=-1)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y


def expand_box(box, rate=5):
    y_min = int(box[1])
    y_max = int(box[3])
    x_min = int(box[0])
    x_max = int(box[2])
    ref_width = x_max - x_min
    x_max = int(float(x_max) + 0.05 * ref_width)  # 0.2
    x_min = int(float(x_min) - 0.1 * ref_width)  # 0.15

    real_width = x_max - x_min

    real_height = real_width / 5 / 2

    y_center = int((y_min + y_max) / 2)

    y_max = y_center + real_height
    y_min = y_center - real_height

    return [x_min, y_min, x_max, y_max]


def get_box_padding(rect, h, w):
    l, t, r, b = 0, 0, 0, 0
    if rect[0] < 0:
        l = -rect[0]
    if rect[1] < 0:
        t = -rect[1]
    if rect[2] > w:
        r = rect[2] - w
    if rect[3] > h:
        b = rect[3] - h
    return [l, t, r, b]


def pre_crop(im_pil, face_rect):
    w, h = im_pil.size
    sq_rect = expand_box(face_rect)
    padding = get_box_padding(sq_rect, h, w)
    transform_list = []
    if set(padding) != {0}:
        transform_list.append(transforms.Pad(tuple(padding), padding_mode='constant'))
        x1, y1, x2, y2 = sq_rect
        _w, _h = x2 - x1, y2 - y1
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = x1 + _w
        y2 = y1 + _h
        crop_rect = (x1, y1, x2, y2)
    else:
        crop_rect = tuple(sq_rect)
    transform_list.append(transforms.Lambda(lambda img: img.crop(crop_rect)))
    trans = transforms.Compose(transform_list)

    im_crop = trans(im_pil.copy())

    return im_crop, padding, sq_rect
