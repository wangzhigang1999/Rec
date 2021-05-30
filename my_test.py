import os

import cv2
import torch
from tqdm import tqdm

from models.experimental import attempt_load
from util import load_image, xywh2xyxy

dir = r"E:\222\ElectricityMeter\training\images"

def test(weights, cuda=False):
    # Initialize/load model and set device
    imgsz = 640
    device = torch.device('cuda:0' if cuda else 'cpu')
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.half()
    # Configure
    model.eval()
    # model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    model(torch.zeros(1, 3, imgsz, imgsz).to(device))  # run once

    for i in tqdm(os.listdir(dir)):

        img = load_image(os.path.join(dir, i))
        img = img.to(device, non_blocking=True)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = torch.unsqueeze(img, 0)

        with torch.no_grad():
            out, _ = model(img, augment=False)  # inference and training outputs
            predn = out[0]
            max_conf, box = 0, []
            for *xyxy, conf, _ in predn.tolist():
                if conf > max_conf:
                    max_conf = conf
                    box = xyxy
                    continue
            box = [int(i) for i in xywh2xyxy(box)]

            left, top, right, bottom = box[0], box[1], box[2], box[3]
            left -= 5
            left = max(left, 0)
            top -= 8
            right += 5
            bottom += 8

            img = cv2.imread(os.path.join(dir, i))
            res = cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            cropImg = res[top:bottom, left:right]

            cv2.imshow("src", res)
            # cv2.imshow("crop", cropImg)
            # cv2.imwrite("{}/{}".format("bar_data", i), cropImg)
            cv2.waitKey(0)


if __name__ == '__main__':
    test(r"best.pt")
