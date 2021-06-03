import os

import cv2


def ttt(path):
    img = cv2.imread(path, 0)
    edges = cv2.Canny(img, 100, 200)  # 参数:图片，minval，maxval,kernel = 3
    # cv2.imshow("sss", edges)
    # cv2.waitKey(0)
    return edges


if __name__ == '__main__':
    dir = "crop"
    all = os.listdir((dir))
    for i in all:
        path = os.path.join(dir, i)
        res = ttt(path)
        cv2.imwrite(path, res)
