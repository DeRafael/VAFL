from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def ROC_curve(label, pred, name):
    # print(name)
    fpr, tpr, threshold = roc_curve(label, pred)
    # print(name)
    roc_auc = auc(fpr, tpr)
    '''
    plt.figure()
    lw = 2
    plt.figure(figsize=(8,6))
    p1, = plt.plot(fpr, tpr, color='k', lw=lw, label='Central ROC curve (area = %0.2f)' % roc_auc)
    p2, = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    # plt.title('Receiver operating characteristic example')
    plt.legend([p1], [name], loc='best', fontsize=15)
    plt.savefig(name + '.eps')
    plt.close()
    '''
    return roc_auc
