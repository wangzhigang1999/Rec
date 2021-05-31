import os

import cv2

dir = "crop"

imgs = os.listdir(dir)

rates = {}
for i in imgs:
    path = os.path.join(dir, i)
    im = cv2.imread(path)
    h, w, _ = im.shape
    rate = int((w / h) * 10) / 10

    try:
        rates[rate] += 1
    except:
        rates[rate] = 1

rates = [(k, rates[k]) for k in sorted(rates.keys())]

for k, v in rates:
    print(k, v)
