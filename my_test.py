import os

import cv2
import torch
from PIL import Image
from tqdm import tqdm

from models.experimental import attempt_load
from util import load_image, xywh2xyxy, pre_crop

dir = r"E:\222\ElectricityMeter\training\images"

w_h_rate = 5.0

from utils.general import non_max_suppression


def test(weights, cuda=True):
    # Initialize/load model and set device
    imgsz = 640
    device = torch.device('cuda:0' if cuda else 'cpu')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.half()
    # Configure
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    for i in tqdm(os.listdir(dir)):
        img = load_image(os.path.join(dir, i))
        img = img.to(device, non_blocking=True)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.unsqueeze(img, 0)

        with torch.no_grad():
            out = model(img, augment=True)[0]

            pred = non_max_suppression(out)
            for _, det in enumerate(pred):
                classes = []
                for *yolo_box, conf, cls in reversed(det):
                    # classes.append(int(cls))
                    crop_and_save(box=yolo_box, i=i)


def crop_and_save(box, i):
    img = Image.open(os.path.join(dir, i))
    res, _, _ = pre_crop(img, box)
    res = res.resize((500, 100))
    res.save("{}/{}".format("crop", i))


def find_best_box(out):
    predn = out[0]
    max_conf, box, cls = 0, [], None
    results = predn.tolist()
    for *xyxy, conf, cls in results:
        if conf > max_conf:
            max_conf = conf
            box = xyxy
            cls = int(cls)
            continue
    box = [int(i) for i in xywh2xyxy(box)]
    return box, cls


def show_cv2(i, box):
    left, top, right, bottom = box[0], box[1], box[2], box[3]
    img = cv2.imread(os.path.join(dir, i))
    res = cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
    cropImg = res[top:bottom, left:right]
    cv2.imshow("src", res)
    cv2.imshow("crop", cropImg)
    cv2.waitKey(0)


if __name__ == '__main__':
    test(r"C:\Users\wanz\PycharmProjects\machine_learning\ckpt\350_best.pt")
