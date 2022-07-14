from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def cm(y_true, y_pred, labels=[0, 1, 2]):
    y_pred = [np.argmax(i, axis=-1)  for i in y_pred]
    return multilabel_confusion_matrix(np.argmax(y_true, axis=-1),  y_pred, labels=labels)
          

def plot_cm(confusion, classes):
    fig = plt.figure(figsize = (14, 8))
    for i, (label, matrix) in enumerate(zip(classes, confusion)):
        plt.subplot(f'23{i+1}')
        labels = [f'not_{label}', label]
        sns.heatmap(matrix, annot = True, square = True, fmt = 'd', cbar = False, cmap = 'Blues', 
                    xticklabels = labels, yticklabels = labels, linecolor = 'black', linewidth = 1)
        plt.title(labels[0])

    plt.tight_layout()
    plt.show()
