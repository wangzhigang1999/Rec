import pickle


def load_pkl(pkl):
    with open(pkl, "rb+")as f:
        return pickle.load(f)


def count():
    res_pkl = "predict.pkl"
    label_pkl = "label.pkl"

    label = load_pkl(label_pkl)
    print(len(label))
    res = load_pkl(res_pkl)
    all = 0
    acc = 0
    wrong = 0
    for k in label.keys():
        try:
            if int(label[k][:-1]) == int(res[k][:-1]):
                acc += 1
            else:
                wrong += 1
                print("key:{}  label:{}  predict:{}".format(k, label[k][:-1], res[k][:-1]))
            all += 1

        except:
            pass
            # wrong += 1
            # print("key:{}  label:{}  predict:{}".format(k, label[k], res[k]))

    print("total:{}   acc:{}   wrong:{}   acc_rate:{}".format(all, acc, wrong, acc / all))

if __name__ == '__main__':
    count()