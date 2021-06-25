import argparse
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from count_acc import count
from models.experimental import attempt_load
from util import load_image, pre_crop, xywh2xyxy
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device

label_pkl = "label.pkl"

parser = argparse.ArgumentParser()
parser.add_argument('--img-size', type=int, default=500, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.01, help='IOU threshold for NMS')
parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--agnostic-nms', default=True, help='class-agnostic NMS')
parser.add_argument('--augment', default=True, help='augmented inference')
parser.add_argument('--save_img', default=True, help='show inference res')
parser.add_argument('--print', default=False, help='show inference res')
parser.add_argument('--use_max_box', default=True, help='whether use max box')
parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
opt = parser.parse_args()

print(opt)

def load_pkl(pkl):
    with open(pkl, "rb+")as f:
        return pickle.load(f)


labels = load_pkl(label_pkl)


def convert(path):
    img = cv2.imread(path)  # BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(path, binary)


def img_preprocess_for_rec(path):
    img = cv2.imread(path, -1)  # BGR (-1 is IMREAD_UNCHANGED)
    img = np.stack((img,) * 3, axis=-1)
    img = letterbox(img, 512, stride=32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img


def init_detect_model(weights):
    imgsz = 640
    device = torch.device('cuda:0')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.half()
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return model


def init_rec_model(weights):
    imgsz = 512
    device = torch.device('cuda:0')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.half()
    model.eval()
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    return model


def find_best_box(out):
    predn = out[0]
    max_conf, box, cls = 0, [], None
    results = predn.tolist()
    for *xyxy, conf, cls in results[0]:
        if conf > max_conf:
            max_conf = conf
            box = xyxy
            continue
    box = [int(i) for i in xywh2xyxy(box)]
    return box


@torch.no_grad()
def detect(model, path):
    img = load_image(path)
    img = img.to(torch.device("cuda:0"), non_blocking=True)
    img = img.half()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = torch.unsqueeze(img, 0)
    out = model(img, augment=True)
    i = path.split("\\")[-1]
    save_path = "test/{}".format(i)
    img = Image.open(path)
    pred = non_max_suppression(out[0], conf_thres=0.25, iou_thres=0.45, max_det=300)
    box = None

    if opt.use_max_box:
        box = find_best_box(out)
    else:
        for _, det in enumerate(pred):
            for *yolo_box, conf, cls in reversed(det):
                box = yolo_box

    if box is None:
        return None

    res, _, _ = pre_crop(img, box)
    res = res.resize((500, 100))
    res.save(save_path)
    return save_path


@torch.no_grad()
def rec(opt, model, path):
    global labels
    device = opt.device
    convert(path)
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
                if opt.save_img:
                    cv2.imwrite(path, original_img)

    for i in sorted(classes.keys()):
        predict += str(classes[i])
        # if opt.print:
        #     print(classes[i], end="")

    # k = int(path.split("/")[-1].split(".")[0])
    #
    # try:
    #     if int(labels[k][:-1]) != int(predict[:-1]):
    #         shutil.copy(path, "error/{}.jpg".format(k))
    # except Exception as e:
    #     pass
    return predict


if __name__ == '__main__':

    detect_model = init_detect_model(r"ckpt/detect/623.pt")
    rec_model = init_rec_model("ckpt/rec/450.pt")

    dir = r"C:\Users\wanz\Desktop\t\ElectricityMeter\training\images"
    all_res = {}
    for i in tqdm(os.listdir(dir)):
        idx = str(i).split(".")[0]
        path = detect(detect_model, os.path.join(dir, i))

        if path is None:
            all_res[int(idx)] = "000000"
            continue

        predict = rec(opt, rec_model, path)
        all_res[int(idx)] = predict

    with open("predict.pkl", "wb+")as f:
        pickle.dump(all_res, f)

    count()
