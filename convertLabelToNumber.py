import os
import pickle

from tqdm import tqdm

res = {}


def convert(path):
    idx = int(path.split("\\")[-1].split(".")[0])
    str = ""
    with open(path) as f:
        for line in f:
            str += line.split(" ")[0]

    res[idx] = str


if __name__ == '__main__':
    label_dir = r"C:\Users\wanz\Desktop\label"

    labels = os.listdir(label_dir)

    for label in tqdm(labels):
        convert(os.path.join(label_dir, label))

    with open("label.pkl", "wb+")as f:
        pickle.dump(res, f)
