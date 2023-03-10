import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from blind_robot.utils.data_utils import int2task

def get_confusion_matrix(targets, predictions, title="plot"):
    """
    Plots the confusion matrix for targets vs model predictions.
    Args
        targets (list): list of target classes
        Preds (list): list of prediction classes
        title (str): title of confusion matrix plot
    """
    y, pred = [], []
    for idx, _ in enumerate(targets):
        y.append(targets[idx])
        pred.append(predictions[idx])

    cm = confusion_matrix(y, pred)

    df_cm = pd.DataFrame(cm, index=int2task, columns=int2task)

    plt.figure(figsize = (12,8))
    plt.title(title) # val_acc 0.81723
    sn.heatmap(df_cm, annot=True)
    plt.xticks(rotation=90)
    plt.show()


