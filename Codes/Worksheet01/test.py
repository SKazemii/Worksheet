import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
from scipy.spatial import distance

def compute_eer(fpr, tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = np.subtract(1, tpr)
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer  # , thresholds[min_index]


model1 = np.load("./Datasets/model1.npy")
model2 = np.load("./Datasets/model2.npy")

THRESHOLDs = [0, 0.2, 0.4, 0.5, 0.77, 0.9, 1.5, 2, 5]
FN = list(); TP = list(); TN = list(); FP = list()
ACC = list(); FDR = list(); FNR = list(); FPR = list(); NPV = list(); PPV = list(); TNR = list(); TPR = list()
# EER = list()

for idx, thresh in enumerate(THRESHOLDs):
    TPM = np.zeros((model1.shape))
    TPM[model1 < thresh] = 1
    TP.append(TPM.sum()/16)
    

    FNM = np.zeros((model1.shape))
    FNM[model1 >= thresh] = 1
    FN.append(FNM.sum()/16)

    FPM = np.zeros((model2.shape))
    FPM[model2 < thresh] = 1
    FP.append(FPM.sum()/16)

    TNM = np.zeros((model2.shape))
    TNM[model2 >= thresh] = 1
    TN.append(TNM.sum()/16)

    # Sensitivity, hit rate, recall, or true positive rate
    # reflects the classifier’s ability to detect members of the positive class (pathological state)
    TPR.append(TP[idx] / (TP[idx]  + FN[idx] ))
    # Specificity or true negative rate
    # reflects the classifier’s ability to detect members of the negative class (normal state)
    TNR.append(TN[idx]  / (TN[idx]  + FP[idx] ))
    # Precision or positive predictive value
    PPV.append(TP[idx]  / (TP[idx]  + FP[idx] ))
    # Negative predictive value
    NPV.append(TN[idx]  / (TN[idx]  + FN[idx] ))
    # Fall out or false positive rate
    # reflects the frequency with which the classifier makes a mistake by classifying normal state as pathological
    FPR.append(FP[idx]  / (FP[idx]  + TN[idx] ))
    # False negative rate
    # reflects the frequency with which the classifier makes a mistake by classifying pathological state as normal
    FNR.append(FN[idx]  / (TP[idx]  + FN[idx] ))
    # False discovery rate
    FDR.append(FP[idx]  / (TP[idx]  + FP[idx] ))
    # Overall accuracy
    ACC.append((TP[idx]  + TN[idx] ) / (TP[idx]  + FP[idx]  + FN[idx]  + TN[idx] ))

EER = compute_eer(FPR, TPR)




print("\n#################################################################################################")
print("#################################################################################################\n")
print("THRESHOLDs:                                                                          {}".format(THRESHOLDs))
print("EER:                                                                                 {}".format(EER))
print("False Positive (FP):                                                                 {}".format(FP))
print("False Negative (FN):                                                                 {}".format(FN))
print("True Positive (TP):                                                                  {}".format(TP))
print("True Negative (TN):                                                                  {}".format(TN))
print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR))
print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR))
print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV))
print("Negative Predictive Value (NPV):                                                     {}".format(NPV))
print(
     "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
        FPR
    )
)
print(
     "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
        FNR
    )
)
print("False Discovery Rate (FDR):                                                          {}".format(FDR))
print("Overall accuracy (ACC):                                                              {}".format(ACC))
print("\n#################################################################################################")
print("\n#################################################################################################")
print("#################################################################################################\n")
print("THRESHOLDs:                                                                          {}".format(THRESHOLDs[4]))
print("EER:                                                                                 {}".format(EER))
print("False Positive (FP):                                                                 {}".format(FP[4]))
print("False Negative (FN):                                                                 {}".format(FN[4]))
print("True Positive (TP):                                                                  {}".format(TP[4]))
print("True Negative (TN):                                                                  {}".format(TN[4]))
print("True Positive Rate (TPR)(Recall):                                                    {}".format(TPR[4]))
print("True Negative Rate (TNR)(Specificity):                                               {}".format(TNR[4]))
print("Positive Predictive Value (PPV)(Precision):                                          {}".format(PPV[4]))
print("Negative Predictive Value (NPV):                                                     {}".format(NPV[4]))
print(
     "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):      {}".format(
        FPR[4]
    )
)
print(
     "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):  {}".format(
        FNR[4]
    )
)
print("False Discovery Rate (FDR):                                                          {}".format(FDR[4]))
print("Overall accuracy (ACC):                                                              {}".format(ACC[4]))
print("\n#################################################################################################")
auc = 1 * np.trapz(TPR, FPR)

plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve, AUC = %.2f'%auc)
plt.legend(loc="lower right")
plt.savefig('AUC.png')

plt.figure()
plt.plot(FPR, FNR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Acceptance Rate')
plt.ylabel('False Rejection Rate')
# plt.title('ROC curve, AUC = %.2f'%auc)
# plt.legend(loc="lower right")
plt.savefig('AUC-1.png')



plt.figure()
plt.plot(THRESHOLDs, FPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='FAR curve', clip_on=False)
plt.plot(THRESHOLDs, FNR, linestyle='--', marker='o', color='navy', lw = 2, label='FRR curve', clip_on=False)
plt.legend(loc="upper right")
plt.savefig('AUC-2.png')

plt.show()