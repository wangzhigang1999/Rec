import cv2
import numpy as np
import torch

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

####################################################

conf_thres = 0.50
iou_thres = 0.01
classes = 10
agnostic_nms = True
max_det = 1000
####################################################

device = torch.device("cuda:0")

net = torch.load("rec_model.pkl")
net = net.to(device)
net.half()
net(torch.zeros(1, 3, 512, 512).to(device).type_as(next(net.parameters())))  # run once


def img_preprocess_for_rec(path):
    img = cv2.imread(path, -1)  # BGR (-1 is IMREAD_UNCHANGED)
    img = np.stack((img,) * 3, axis=-1)
    img = letterbox(img, 512, stride=32)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def post_process_rec(input, pre, path):
    img = input
    im0 = cv2.imread(path)
    pred = non_max_suppression(pre, conf_thres, iou_thres, 10, agnostic_nms, max_det)

    predict = ""
    classes = {}

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                x = xyxy[0]
                classes[int(x)] = c

    for i in sorted(classes.keys()):
        predict += str(classes[i])
        print(classes[i], end="")


if __name__ == '__main__':
    input = img_preprocess_for_rec(r"C:\Users\wanz\PycharmProjects\machine_learning\images\2.JPG")
    res = net(input)[0]
    post_process_rec(input, res, r"C:\Users\wanz\PycharmProjects\machine_learning\images\2.JPG")
    # print(res)
