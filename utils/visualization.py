import matplotlib.pyplot as plt
import seaborn as sns

from config import data_cfg


def show_batch(batch, nrows=4, ncols=4, figsize=2):
    assert len(batch['target']) == ncols * nrows
    reversed_label_mapping = dict(map(lambda x: (x[1], x[0]), data_cfg.label_mapping.items()))
    figure, axs = plt.subplots(nrows, ncols, figsize=(figsize * ncols, figsize * nrows))

    for n, (img, label) in enumerate(zip(batch['image'], batch['target'])):
        i, j = n // ncols, n % ncols
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].set_title(f'Class {reversed_label_mapping[int(label)]}')
        axs[i, j].set_axis_off()

    plt.show()


def plot_confusion_matrix(cm, title, class_labels=None):
    plt.clf(), plt.cla()
    sns.heatmap(cm, annot=True, fmt=".4f", cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    return plt.gcf()
