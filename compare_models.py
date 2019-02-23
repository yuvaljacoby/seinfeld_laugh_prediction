import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def compare_models_roc_curve(y_trues, y_hats, labels, plot=True, out_dir='./'):
    """
    Prints roc curve to compare multiple models

    :param y_true: list of true labels
    :param y_hats: list of models predictions,
        each cell in the list is a list of probabilities with length len(y_true)
    :param labels: list of size len(y_hats) with label for the model
    :param plot: whether to plot the ROC curve or not (and just return auc).
    :return: AUC for each model + print the plot
    """
    auc = {}

    for i, (y_true, y_hat) in enumerate(zip(y_trues, y_hats)):
        fpr, tpr, _ = roc_curve(y_true, y_hat)

        if plot:
            plt.step(fpr, tpr, alpha=0.5, label=labels[i])

        auc[labels[i]] = roc_auc_score(y_true, y_hat)

    if plot:
        y_hat_random = np.random.random(len(y_trues[0]))
        fpr_rand, tpr_rand, _ = roc_curve(y_trues[0], y_hat_random)
        plt.step(fpr_rand, tpr_rand, color='b', alpha=0.5, label='random')

        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.legend()
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig('%s/figures/roc.png'%out_dir)
        plt.show()

    return auc


def calc_best_threshold(y_trues, y_hats, labels):
    """
    For each label calculate the threshold when tpr and fpr has the same importance.
    i.e. take the threshold that minimizes (TPR + FPR)

    :param y_true: labeled data
    :param y_hats: list of models predictions,
        each cell in the list is a list of probabilities with length len(y_true)
    :param labels: list of size len(y_hats) with label for the model

    :return: dict, keys are the labels and values the best threshold
    """
    # Convert y_hats to list of lists if it's not already in this format
    if not (isinstance(y_hats, list) or isinstance(y_hats[0], list)):
        y_hats = [y_hats]

    best_thresholds = {}
    for i, (y_true, y_hat) in enumerate(zip(y_trues, y_hats)):
        fpr, tpr, thresholds = roc_curve(y_true, y_hat)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        best_thresholds[labels[i]] = optimal_threshold

    return best_thresholds


def plot_confusion_matrix(y_trues, y_hats, labels, thresholds=None, out_dir='./'):
    """
    For each label calculate the threshold when TPR and FPR has the same importance.

    :param y_true: labeled data
    :param y_hats: list of models predictions,
        each cell in the list is a list of probabilities with length len(y_true)
    :param labels: list of size len(y_hats) with label for the model
    :param thresholds: dict key is the label value the threshold to discretized y_hat
            if none using best
    :return: dict, keys are the labels and values the best threshold
    """

    if not thresholds:
        thresholds = calc_best_threshold(y_trues, y_hats, labels)

    fig, axs = plt.subplots(len(y_hats), 1, sharex=True)
    fig.set_figheight(7)

    ticks_labels = ['not funny', 'funny']
    for i, (y_true, y_hat) in enumerate(zip(y_trues, y_hats)):
        y_hat_binary = y_hat >= thresholds[labels[i]]
        cm = confusion_matrix(y_true, y_hat_binary)
        # TODO: Not sure if to divide by axis 0 or 1
        cm_norm = cm.astype('float') / cm.sum(axis=0) * 100
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        l = np.asarray(["{0} \n({1:.2f}%)".format(total, prec)
                        for total, prec in
                        zip(cm.flatten(), cm_norm.flatten())]).reshape(2, 2)

        if len(y_hats) > 1:
            curent_ax = axs[i]
        else:
            curent_ax = axs

        sns.heatmap(cm, annot=l, fmt="", ax=curent_ax, xticklabels=ticks_labels, yticklabels=ticks_labels)

        curent_ax .set_title(labels[i])
        # axs[i].set_xlabel('Predicted label')
        curent_ax .set_ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    plt.plot()
    plt.savefig('%s/figures/confusion_matrix.png'%out_dir)
    plt.show()
