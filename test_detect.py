import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from models.experimental import attempt_load
from util import load_image, pre_crop
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device


def img_preprocess_for_rec(path):
    img = cv2.imread(path, -1)  # BGR (-1 is IMREAD_UNCHANGED)
    img = np.stack((img,) * 3, axis=-1)
    img = letterbox(img, 512, stride=32)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def init_rec_model(weight):
    device = torch.device("cuda:0")

    # Load model
    model = torch.load(weight, map_location=device)  # load FP32 model
    model.half()  # to FP16
    cudnn.benchmark = True
    model(torch.zeros(1, 3, 512, 512).to(device).type_as(next(model.parameters())))  # run once

    return model


def init_detect_model(weights):
    imgsz = 640
    device = torch.device('cuda:0')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.half()
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return model


@torch.no_grad()
def detect(model, path):
    img = load_image(path)
    img = img.to(torch.device("cuda:0"), non_blocking=True)
    img = img.half()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.unsqueeze(img, 0)
    out = model(img, augment=True)[0]

    pred = non_max_suppression(out)
    for _, det in enumerate(pred):
        for *yolo_box, conf, cls in reversed(det):
            img = Image.open(path)
            res, _, _ = pre_crop(img, yolo_box)
            res = res.resize((500, 100))
            res.save("./test.jpg")


@torch.no_grad()
def rec(opt, model, path):
    save_img = True
    device = opt.device

    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    img = img_preprocess_for_rec(path)
    original_img = cv2.imread(path, -1)  # BGR (-1 is IMREAD_UNCHANGED)
    original_img = np.stack((original_img,) * 3, axis=-1)

    classes = {}
    predict = ""
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_img.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                plot_one_box(xyxy, original_img, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                classes[int(xyxy[0])] = c

    if save_img:
        cv2.imshow("inference", original_img)
        cv2.waitKey(0)

    for i in sorted(classes.keys()):
        predict += str(classes[i])
        print(classes[i], end="")
    return predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=500, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.01, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save_img', action='store_true', help='show inference res')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    opt = parser.parse_args()

    detect_model = init_detect_model(r"ckpt/detect/350_best.pt")
    rec_model = init_rec_model("ckpt/rec_model.pkl")

    detect(detect_model, "full_test.JPG")

    rec(opt, rec_model, r"C:\Users\wanz\PycharmProjects\machine_learning\images\2.JPG")
