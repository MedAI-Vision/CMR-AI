import os
import numpy as np
import pickle
import shutil

import matplotlib.pyplot as plt
from sklearn import metrics
np.set_printoptions(threshold=1e5)

disease_map = {'HCM': 0, 'DCM': 1, 'CAD': 2, 'ARVC': 3, 'PAH': 4, 'Myocarditis': 5,
               'RCM': 6, 'Ebstein’s Anomaly': 7, 'HHD': 8, 'CAM': 9, 'LVNC': 10}
disease_map_reverse = dict(zip(disease_map.values(),disease_map.keys()))

exp_file = dict()
exp_file['cine'] = 'data/sax_4ch_cine_fusion_11cls/20ep/ex/'
exp_file['lge'] = 'data/sax_lge_11cls/ex/'
exp_file['fusion'] = 'data/sax_4ch_lge_fusion_11cls/20ep/ex/'

def get_roc_stat_muti(exp_file:dict, label_dict:dict):
    """
    exp_file: `{'cine': cine root, 'lge': lge root, 'fusion': cine+lge root}`\n
    label: `{'disease name': label}`
    \n
    return: fpr_list, tpr_list, roc_auc_list
    each element (dict) of list specifies a class in label_dict
    """
    pred_dict = {}
    real_dict = {}
    scores_dict = {}

    for mod in ['cine', 'lge', 'fusion']:
        root = exp_file[mod]
        pred_dict[mod] = np.load(os.path.join(root, 'Predicted_Value.npy'))
        real_dict[mod] = np.load(os.path.join(root, 'True_Value.npy'))
        scores_dict[mod] = np.load(os.path.join(root, 'scores.npy'))

#    assert real_dict['cine'] == real_dict['lge'] and real_dict['cine'] == real_dict['fusion'],\
#        'please assure using the same testing set'

    roc_auc_list = []
    fpr_list = []
    tpr_list = []
    for index in range(len(label_dict)):
        roc_auc_cls = {}
        fpr_cls = {}
        tpr_cls = {}
        for mod in ['cine', 'lge', 'fusion']:
            scores = scores_dict[mod]
            real = real_dict[mod]
            newscore=scores[:,index]
            fpr,tpr,thresholds=metrics.roc_curve(real,newscore,pos_label=index)
            roc_auc=metrics.auc(fpr,tpr)

            roc_auc_cls[mod] = roc_auc
            fpr_cls[mod] = fpr
            tpr_cls[mod] = tpr

        fpr_list.append(fpr_cls)
        tpr_list.append(tpr_cls)
        roc_auc_list.append(roc_auc_cls)

    return fpr_list, tpr_list, roc_auc_list

fpr_list, tpr_list, roc_auc_list = get_roc_stat_muti(exp_file, disease_map)

def get_multi_roc_graph(path, fpr_list, tpr_list, roc_auc_list, label_dict_re):
    try:
        os.makedirs(path)
    except:
        pass
    for i in range(len(label_dict_re)):
        fpr = fpr_list[i]
        tpr = tpr_list[i]
        roc_auc = roc_auc_list[i]
        plt.figure(figsize=(5,5))
        plt.plot(fpr['cine'], tpr['cine'], lw=1.2, label='{}, AUC = {:.3f}'.format('cine', roc_auc['cine']), color='#C45AEC')
        plt.plot(fpr['lge'], tpr['lge'], lw=1.2, label='{}, AUC = {:.3f}'.format('LGE', roc_auc['lge']), color='#ED7D31')
        plt.plot(fpr['fusion'], tpr['fusion'], lw=1.2, label='{}, AUC = {:.3f}'.format('cine + LGE', roc_auc['fusion']), color='#00B0F0')
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.legend(loc="lower right",frameon=False)
        plt.xlabel('1 - Specificity', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Sensitivity', fontsize=12, fontname='Times New Roman')
        plt.xlim((-0.001,1))
        plt.ylim((0,1))
        plt.xticks([0,0.2,0.4,0.6,0.8,1.0],['0·0','0·2','0·4','0·6','0·8','1·0'])
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0],['0·0','0·2','0·4','0·6','0·8','1·0'])
        plt.plot([0, 1], [0, 1],color="#506267", linestyle='--', dashes=(2, 3))
        plt.title(f'{label_dict_re[i]}')
        ax=plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(path, f'{label_dict_re[i]}_multi_roc.jpg'), dpi=200)

get_multi_roc_graph(path='Legacy/FinalizeExp/ex_ROC', fpr_list=fpr_list, tpr_list=tpr_list, roc_auc_list=roc_auc_list, label_dict_re=disease_map_reverse)