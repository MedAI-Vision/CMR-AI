# calculate auc, f1-score, specificity, sensitivity and their confidence intervals by bootstrap method
# for multilabel tasks
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# INPUTS scores:[n, 11], pred:[n,], label[n,], different tasks only need to modify the following three lines
work_dir = 'data/sax_lge_11cls/revise/'
scores = np.load(work_dir + 'scores.npy')
pred = np.load(work_dir + 'Predicted_Value.npy')
label = np.load(work_dir + 'True_Value.npy')

print(label.shape)
disease_map = {'HCM': 0, 'DCM': 1, 'CAD': 2, 'ARVC': 3, 'PAH': 4, 'Myocarditis': 5,
               'RCM': 6, 'Ebstein’s Anomaly': 7, 'HHD': 8, 'CAM': 9, 'LVNC': 10}
disease_map_reverse = dict(zip(disease_map.values(), disease_map.keys()))

def plot_confusion_matrix(cm: np.ndarray, save_path='', title='Confusion matrix', cmap=None,normalize=True, show=True):

    """
    Plot the confusion matrix given the confusion matrix and the labels.
    @param cm - the confusion matrix
    @param target_names - the labels
    @param save_path - the path to save the plot
    @param title - the title of the plot
    @param cmap - the color map
    @param normalize - whether to normalize the confusion matrix
    @param show - whether to show the plot
    @returns 0
    """

    font = {'weight': 'normal',
            'color': 'white',
            'size': 12}
    fontb = {'weight': 'normal',
             'color': 'black',
             'size': 12}
    disease_map = {'HCM': 0, 'DCM': 1, 'CAD': 2, 'LVNC': 10, 'RCM': 6, 'CAM': 9, 'HHD': 8,
                   'Myocarditis': 5, 'ARVC': 3, 'PAH': 4, 'Ebstein’s Anomaly': 7}

    target_names = disease_map.keys()
    cm[[3, 4, 5, 6, 7, 8, 9, 10], :] = cm[[10, 6, 9, 8, 5, 3, 4, 7], :]
    cm[:, [3, 4, 5, 6, 7, 8, 9, 10]] = cm[:, [10, 6, 9, 8, 5, 3, 4, 7]]

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    confusion = np.transpose(cm)
    if cmap is None:
        cmap = 'Blues'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0
    print(cm.shape)

    plt.figure(figsize=(10, 9))
    plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    cb1 = plt.colorbar(fraction=0.045)
    cb1.ax.tick_params(labelsize=12)
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=12, fontname='ARIAL', rotation=30, ha='right')
        plt.yticks(tick_marks, target_names, fontsize=12, fontname='ARIAL')
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    thresh = 0.5 * np.max(confusion) / np.sum(confusion)
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            if confusion[first_index][second_index] / np.sum(confusion) <= thresh:
                plt.text(first_index, second_index,
                         confusion[first_index][second_index], fontdict=fontb)
            else:
                plt.text(first_index, second_index,
                         confusion[first_index][second_index], fontdict=font)
    plt.tight_layout()
    plt.ylabel('True label', size=15)
    plt.xlabel('Predicted label', size=15)
    if save_path != '':
        plt.savefig(os.path.join(save_path, 'CM-Fuwai&Ex.jpg'), bbox_inches='tight', dpi=500)
    if show:
        plt.show()
    return 0

# confusion matrix
conf_mat = confusion_matrix(y_true=label, y_pred=pred)
print(conf_mat)

plot_confusion_matrix(conf_mat, save_path=work_dir)
# np.save((work_dir +'Confusion_Matrix.npy'), conf_mat)

# diseases order list
diseases = ['HCM', 'DCM', 'CAD', 'ARVC', 'PAH', 'myocarditis', 'RCM', 'CHD', 'HHD', 'Cardiac', 'LVNC']

def AUC(data):
    """
    Calculate AUC（for bootstrap）
    :param data: data[:, 0] = score, data[:, 1] = label, data[:, 2] = index
    :return: auc score
    """
    pos_index = int(data[0, 2])
    fpr, tpr, thersholds = roc_curve(data[:, 1].astype(int).tolist(), data[:, 0].tolist(), pos_label=pos_index)
    roc_auc = auc(fpr, tpr)
    return roc_auc

# def f1_scores(data):
#     """
#     Calculate f1 score（for bootstrap）
#     :param data: data[:, 0] = score, data[:, 1] = label, data[:, 2] = index
#     :return: f1 score
#     """
#     pos_index = int(data[0, 2])
#     tp = conf_mat[pos_index][pos_index]
#     fp = np.sum(conf_mat[:, pos_index]) - tp
#     fn = np.sum(conf_mat[pos_index]) - tp
#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1_cls = 2 * precision * recall / (precision + recall)
#     return f1_cls
def f1_scores(data):
    """
    Calculate f1 score（for bootstrap）
    :param data: data[:, 0] = pred, data[:, 1] = label, data[:, 2] = index
    :return: f1 score
    """
    label_f1 = data[:, 1]
    pred_f1 = data[:, 0]
    conf_mat = confusion_matrix(y_true=label_f1, y_pred=pred_f1)
    pos_index = int(data[0, 2])
    tp = conf_mat[pos_index][pos_index]
    fp = np.sum(conf_mat[:, pos_index]) - tp
    fn = np.sum(conf_mat[pos_index]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_cls = 2 * precision * recall / (precision + recall)
    return f1_cls

def f1_scores_all(data):
    label_f1 = data[:, 1]
    pred_f1 = data[:, 0]
    f1_cls = f1_score(label_f1, pred_f1, average='weighted')
    return f1_cls

def get_specificity(data):
    """
    Calculate specificity（for bootstrap）
    :param data: data[:, 0] = score, data[:, 1] = label, data[:, 2] = index
    :return: specificity (when sensitivity=0.9)
    """
    pos_index = int(data[0, 2])
    fpr, tpr, thersholds = roc_curve(data[:, 1].astype(int).tolist(), data[:, 0].tolist(), pos_label=pos_index)
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    for i in range(len(tpr)):
        if tpr[i] == 0.9:
            return 1 - fpr[i]
        elif tpr[i] < 0.9 and tpr[i + 1] > 0.9:
            a = 0.9 - tpr[i]
            b = tpr[i + 1] - 0.9
            fpr_i = a / (a + b) * fpr[i + 1] + b / (a + b) * fpr[i]
            return 1 - fpr_i


def get_sensitivity(data):
    """
    Calculate sensitivity（for bootstrap）
    :param data: data[:, 0] = score, data[:, 1] = label, data[:, 2] = index
    :return: sensitivity (when specificity=0.9)
    """
    pos_index = int(data[0, 2])
    fpr, tpr, thersholds = roc_curve(data[:, 1].astype(int).tolist(), data[:, 0].tolist(), pos_label=pos_index)
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    for i in range(len(fpr)):
        if fpr[i] == 0.1:
            return tpr[i]
        elif fpr[i] < 0.1 and fpr[i + 1] > 0.1:
            a = 0.1 - fpr[i]
            b = fpr[i + 1] - 0.1
            tpr_i = a / (a + b) * tpr[i + 1] + b / (a + b) * tpr[i]
            return tpr_i


def bootstrap(data, B, c, func):
    """
    Calculate bootstrap confidence interval
    :param data: array, sample data
    :param B: Sampling times normally, B>=1000
    :param c: Confidence level, for example, 0.95
    :param func: estimator
    :return: upper and lower bounds of bootstrap confidence interval
    """
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)

    a = 1 - c
    k1 = int(B * a / 2)
    k2 = int(B * (1 - a / 2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]

    return lower, higher



# data_f1_all = np.zeros((scores.shape[0], 2))
# data_f1_all[:, 0] = pred
# data_f1_all[:, 1] = label
# f1_score_all = f1_scores_all(data_f1_all)
# result_f1_all = bootstrap(data_f1_all, 1000, 0.95, f1_scores_all)
# print('Weighted F1-Score: {:.3f} ({:.3f}, {:.3f})'.format(np.round(f1_score_all, 3), np.round(result_f1_all, 3)[0],
#                                                      np.round(result_f1_all, 3)[1]))



weighted_auroc = 0
weighted_F1 = 0

for index in range(scores.shape[-1]):
    data = np.zeros((scores.shape[0], 3))
    score = scores[:, index]
    data[:, 0] = score
    data[:, 1] = label
    data[:, 2] = index

    data_f1 = np.zeros((scores.shape[0], 3))
    data_f1[:, 0] = pred
    data_f1[:, 1] = label
    data_f1[:, 2] = index

    # Input must be a data
    roc_auc = AUC(data)
    f1_score_ = f1_scores(data_f1)
    sensitivity = get_sensitivity(data)
    specificity = get_specificity(data)

    # bootstrap for 1000, calculate 95% CI
    result = bootstrap(data, 1000, 0.95, AUC)
    result2 = bootstrap(data_f1, 1000, 0.95, f1_scores)
    result3 = bootstrap(data, 1000, 0.95, get_sensitivity)
    result4 = bootstrap(data, 1000, 0.95, get_specificity)

    # Output
    print('--------------------------------------------------')
    print('Disease: ', diseases[index])
    print('AUROC: {:.3f} ({:.3f}, {:.3f})'.format(np.round(roc_auc, 3), np.round(result, 3)[0], np.round(result, 3)[1]))
    print('F1-Score: {:.3f} ({:.3f}, {:.3f})'.format(np.round(f1_score_, 3), np.round(result2, 3)[0],
                                                     np.round(result2, 3)[1]))
    print('Specificity: {:.3f} ({:.3f}, {:.3f})'.format(np.round(specificity, 3), np.round(result4, 3)[0],
                                                        np.round(result4, 3)[1]))
    print('Sensitivity: {:.3f} ({:.3f}, {:.3f})'.format(np.round(sensitivity, 3), np.round(result3, 3)[0],
                                                        np.round(result3, 3)[1]))

    weight = np.sum(label == index) / scores.shape[0]
    print(disease_map_reverse[index], np.sum(label == index))

    weighted_auroc += weight*roc_auc
    weighted_F1 += weight*f1_score_
print('--------------------------------------------------')
print('weighted_auroc: ', weighted_auroc)
print('weighted_F1: ', weighted_F1)