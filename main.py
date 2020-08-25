import glob
from model import predict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import normalize
import numpy as np


def get_result():
    tol = 0.80
    pictures = glob.glob("./test/**/*.png")
    res = [predict(path)[0][0] for path in tqdm(pictures)]
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(700):
        if res[i] > tol:
            tp += 1
        else:
            fn += 1
    for i in range(700, 1400):
        if res[i] > tol:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn, len(pictures)


def plotMatrix(cm):
    cm = normalize(cm)
    display_labels = ["with", "without"]
    include_values = True
    cmap = "Blues"
    ax = None
    xticks_rotation = 'horizontal'

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels)

    disp = disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)
    disp.ax_.set_title('Normalised Confusion Matrix')

    plt.savefig('fig.png')


def analysis():
    tp, tn, fp, fn, total = get_result()
    cm = [[tp, fn], [fp, tn]]
    cm = np.asarray(cm)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 / ((1/recall) + (1/precision))
    print(f"Accuracy = {accuracy}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F_Score = {f_score}")
    plotMatrix(cm)


if __name__ == "__main__":
    analysis()
