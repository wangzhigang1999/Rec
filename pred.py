from __future__ import print_function, unicode_literals

import json
import os

import cv2
from torch.backends import cudnn
from tqdm import tqdm

from full_detect import init_detect_model, init_rec_model, opt, rec, detect

detect_model = init_detect_model(r"ckpt/detect/623.pt")
rec_model = init_rec_model("ckpt/rec/450.pt")

cudnn.benchmark = True


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def read_img(img_name, base_path, set_name):
    img_rgb_path = os.path.join(base_path, set_name, 'images', img_name)
    _assert_exist(img_rgb_path)
    return cv2.imread(img_rgb_path)


def dump(pred_out_path, img_list, pred_list):
    """ Save predictions into a json file. """
    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump([img_list, pred_list], fo)
    print('Dumped %d predictions to %s' % (len(pred_list), pred_out_path))


def pred_template(img):
    """ 根据给定电表图片，预测电表读数。
        img: input RGB image.
        return: string, e.g. '00001'.
    """
    path = detect(detect_model, img)

    if path is None:
        return "000000"

    predict = rec(opt, rec_model, path)
    # the prediction's data type should be string, e.g. '00001'.
    predict = str(int(predict))

    return predict


def main(base_path, pred_out_path, pred_func, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'

    # init output containers
    pred_list = list()

    img_list = os.listdir(os.path.join(base_path, set_name, 'images'))

    # iterate over the dataset once
    for img_name in tqdm(img_list):
        # use some algorithm for prediction
        pred = pred_func(os.path.join(base_path, set_name, "images", img_name))

        pred_list.append(pred)

    # dump results
    dump(pred_out_path, img_list, pred_list)


if __name__ == '__main__':
    # 数据集所在的目录，例如： D:/ElectricityMeter
    base_path = r'C:\Users\wanz\Desktop\t\ElectricityMeter'
    # 保存的预测结果文件的路径，例如： D:/pred.json
    out = 'pred.json'
    # 调用main函数，对测试集进行预测，并保存预测结果。
    main(
        base_path,
        out,
        pred_func=pred_template,
        set_name='training'
    )
