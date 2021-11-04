import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import config as cfg


def compute_eer(fpr, tpr):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer  # , thresholds[min_index]


# lda all
lda_all_fpr = [
    0,
    0.00346488,
    0.00699157,
    0.01189598,
    0.01642858,
    0.01944202,
    0.0229829,
    0.02927221,
    0.03985026,
    0.0512665,
    0.06263582,
    0.07529578,
    0.09152656,
    0.12805981,
    0.18370561,
    0.24161591,
    0.29816065,
    0.4093789,
    0.6927263,
    1,
]
lda_all_tpr = [
    0,
    0.03631916,
    0.07324561,
    0.10748988,
    0.1483637,
    0.21437247,
    0.28949055,
    0.36196019,
    0.4152834,
    0.46998988,
    0.532861,
    0.60480769,
    0.67950405,
    0.73663968,
    0.77952092,
    0.83060054,
    0.90620783,
    0.9729251,
    0.98076923,
    0.98076923,
]
lda_all_auc = 0.884


# knn all
knn_all_fpr = [
    0,
    8.05789318e-04,
    1.94602041e-03,
    4.33630191e-03,
    6.92625670e-03,
    9.30114674e-03,
    1.23216475e-02,
    1.67777268e-02,
    2.14083201e-02,
    2.64648589e-02,
    3.19137598e-02,
    3.91491983e-02,
    4.70756916e-02,
    5.72765273e-02,
    7.45318071e-02,
    1.09468773e-01,
    1.83636825e-01,
    3.61589789e-01,
    6.70816585e-01,
    1.00000000e00,
]
knn_all_tpr = [
    0,
    0.04326923,
    0.08734818,
    0.13259109,
    0.18822537,
    0.25217611,
    0.31445682,
    0.35921053,
    0.40863698,
    0.4613529,
    0.51136977,
    0.55337382,
    0.59532726,
    0.64245951,
    0.6930668,
    0.73846154,
    0.77990891,
    0.83225371,
    0.90580297,
    0.98076923,
]
knn_all_auc = 0.841
# knn_all_tpr = np.array(knn_all_tpr)
# knn_all_fpr = np.array(knn_all_fpr)
# knn_all_eer = (compute_eer(knn_all_fpr, knn_all_tpr))
# svm all
svm_all_fpr = [
    0.00000000e00,
    2.42938149e-04,
    5.09698943e-04,
    2.90633394e-03,
    6.03989481e-03,
    9.62559552e-03,
    1.37084051e-02,
    1.77638726e-02,
    2.66294050e-02,
    3.78973239e-02,
    5.17271528e-02,
    6.86604277e-02,
    9.04647195e-02,
    1.26549236e-01,
    1.99780694e-01,
    2.84253856e-01,
    3.91512504e-01,
    5.35876518e-01,
    7.67753964e-01,
    1.00000000e00,
]
svm_all_tpr = [
    0,
    0.07397099,
    0.1474865,
    0.21465924,
    0.29205466,
    0.36840418,
    0.43402497,
    0.49967949,
    0.55502699,
    0.60573549,
    0.65728745,
    0.71929825,
    0.78567814,
    0.83333333,
    0.85921053,
    0.90037112,
    0.94807692,
    0.98011134,
    0.98076923,
    0.98076923,
]
svm_all_auc = 0.913
# rfc all
rfc_all_fpr = [
    0,
    0.00139211,
    0.00278421,
    0.00417632,
    0.00556843,
    0.00696054,
    0.00835264,
    0.00974475,
    0.01113686,
    0.01252897,
    0.06516054,
    0.16903159,
    0.27290265,
    0.3767737,
    0.48064475,
    0.5845158,
    0.68838685,
    0.7922579,
    0.89612895,
    1.0,
]
rfc_all_tpr = [
    0,
    0.03518893,
    0.07037787,
    0.1055668,
    0.14075574,
    0.17594467,
    0.2111336,
    0.24632254,
    0.28151147,
    0.3167004,
    0.36831984,
    0.43636977,
    0.5044197,
    0.57246964,
    0.64051957,
    0.7085695,
    0.77661943,
    0.84466937,
    0.9127193,
    0.98076923,
]
rfc_all_auc = 0.651
plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(rfc_all_fpr, rfc_all_tpr, marker=".", label="RFC AUC=" + str(rfc_all_auc))
plt.plot(svm_all_fpr, svm_all_tpr, marker=".", label="SVM AUC=" + str(svm_all_auc))
plt.plot(knn_all_fpr, knn_all_tpr, marker=".", label="KNN AUC=" + str(knn_all_auc))
plt.plot(lda_all_fpr, lda_all_tpr, marker=".", label="LDA AUC=" + str(lda_all_auc))

# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(label="ROC curve - all features", fontsize=20)
# show the legend
plt.legend()
# show the plot
plt.savefig(os.path.join(cfg.fig_dir, "roc_curve_all " + cfg.result_name_file + ".png"))


# lda temp
lda_temp_fpr = [
    0,
    0.00170104,
    0.00417543,
    0.00908488,
    0.01259057,
    0.01637325,
    0.02211147,
    0.02764267,
    0.03348209,
    0.04554712,
    0.0599323,
    0.07307306,
    0.09706056,
    0.13548641,
    0.19206663,
    0.24906273,
    0.2851768,
    0.35633226,
    0.66272167,
    1.0,
]
lda_temp_tpr = [
    0,
    0.02125506,
    0.04331984,
    0.08883266,
    0.15877193,
    0.23448043,
    0.29649123,
    0.36280364,
    0.42825574,
    0.48367072,
    0.534278,
    0.5965081,
    0.6655027,
    0.72705803,
    0.76300607,
    0.80111336,
    0.88324899,
    0.96811741,
    0.97935223,
    0.98076923,
]
lda_temp_auc = 0.881
# knn temp
knn_temp_fpr = [
    0,
    0.00122931,
    0.00277812,
    0.00812697,
    0.01554954,
    0.02243837,
    0.02720261,
    0.03385671,
    0.04277275,
    0.05714547,
    0.07136258,
    0.08429237,
    0.09425758,
    0.11153799,
    0.13534432,
    0.17110815,
    0.23158971,
    0.38347677,
    0.6813935,
    1.0,
]
knn_temp_tpr = [
    0,
    0.0330803,
    0.0682861,
    0.10976721,
    0.15666329,
    0.2097166,
    0.27613023,
    0.34642375,
    0.41825236,
    0.46449055,
    0.50985155,
    0.55934548,
    0.61610999,
    0.66388327,
    0.71406883,
    0.76071188,
    0.78675776,
    0.83458165,
    0.90686572,
    0.98076923,
]
knn_temp_auc = 0.819
# svm temp
svm_temp_fpr = [
    0,
    0.00101436,
    0.00314911,
    0.01109623,
    0.01728229,
    0.02147265,
    0.0292263,
    0.03743818,
    0.04657926,
    0.05994691,
    0.07377756,
    0.08757354,
    0.10513225,
    0.13558094,
    0.19334357,
    0.26007728,
    0.30680605,
    0.39000137,
    0.68280748,
    1.0,
]
svm_temp_tpr = [
    0,
    0.05246289,
    0.10305331,
    0.14053644,
    0.19875169,
    0.28599865,
    0.36266869,
    0.42226721,
    0.47329622,
    0.52923414,
    0.5807861,
    0.63240553,
    0.69308367,
    0.75220985,
    0.78822537,
    0.82818826,
    0.90134953,
    0.97263833,
    0.98076923,
    0.98076923,
]
svm_temp_auc = 0.883
# rfc temp
rfc_temp_fpr = [
    0,
    0.00151549,
    0.00303099,
    0.00454648,
    0.00606198,
    0.00757747,
    0.00909297,
    0.01058944,
    0.0120764,
    0.01356336,
    0.06020326,
    0.15199609,
    0.24378893,
    0.34156393,
    0.45130328,
    0.56104262,
    0.67078197,
    0.78052131,
    0.89026066,
    1.0,
]
rfc_temp_tpr = [
    0,
    0.02557355,
    0.0511471,
    0.07672065,
    0.1022942,
    0.12786775,
    0.1534413,
    0.17921727,
    0.20509447,
    0.23097166,
    0.27925101,
    0.34993252,
    0.42061404,
    0.49473684,
    0.57574224,
    0.65674764,
    0.73775304,
    0.81875843,
    0.89976383,
    0.98076923,
]
rfc_temp_auc = 0.608

plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(rfc_temp_fpr, rfc_temp_tpr, marker=".", label="RFC AUC=" + str(rfc_temp_auc))
plt.plot(svm_temp_fpr, svm_temp_tpr, marker=".", label="SVM AUC=" + str(svm_temp_auc))
plt.plot(knn_temp_fpr, knn_temp_tpr, marker=".", label="KNN AUC=" + str(knn_temp_auc))
plt.plot(lda_temp_fpr, lda_temp_tpr, marker=".", label="LDA AUC=" + str(lda_temp_auc))

# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(label="ROC curve - Temperal features", fontsize=20)

# show the legend
plt.legend()
# show the plot
plt.savefig(
    os.path.join(cfg.fig_dir, "roc_curve_temp " + cfg.result_name_file + ".png")
)


# lda spec
lda_spec_fpr = [
    0,
    0.00283116,
    0.0064115,
    0.01321322,
    0.0187426,
    0.02428415,
    0.03182482,
    0.04050144,
    0.05347631,
    0.0664764,
    0.0799559,
    0.09912992,
    0.11618094,
    0.1318035,
    0.1672457,
    0.2417818,
    0.31335434,
    0.40605123,
    0.68290025,
    1.0,
]
lda_spec_tpr = [
    0,
    0.01503036,
    0.03289474,
    0.06356275,
    0.11420378,
    0.19996626,
    0.28395749,
    0.35942982,
    0.41648111,
    0.47432524,
    0.54210526,
    0.60524629,
    0.66214575,
    0.7265857,
    0.7807861,
    0.81864035,
    0.88645412,
    0.96508097,
    0.97955466,
    0.98076923,
]
lda_spec_auc = 0.870
# knn spec
knn_spec_fpr = [
    0.00000000e00,
    9.95776224e-04,
    2.10624021e-03,
    4.81121666e-03,
    8.66028064e-03,
    1.18393828e-02,
    1.45667536e-02,
    1.85846611e-02,
    2.23566805e-02,
    2.82821393e-02,
    3.50110088e-02,
    4.18366609e-02,
    4.69692100e-02,
    5.57816079e-02,
    7.21523740e-02,
    1.11469708e-01,
    1.83686048e-01,
    3.60502001e-01,
    6.75014367e-01,
    1.00000000e00,
]
knn_spec_tpr = [
    0,
    0.03252362,
    0.06449055,
    0.09451754,
    0.13065115,
    0.18363698,
    0.24423077,
    0.29586707,
    0.35575236,
    0.40359312,
    0.44419703,
    0.48022942,
    0.53437922,
    0.59450067,
    0.65113023,
    0.68922065,
    0.72764845,
    0.79665992,
    0.88830972,
    0.98076923,
]
knn_spec_auc = 0.811
# svm spec
svm_spec_fpr = [
    0,
    0.03834033,
    0.07668066,
    0.11536235,
    0.15409636,
    0.19287648,
    0.23292721,
    0.27372444,
    0.31323721,
    0.3518974,
    0.39096828,
    0.43106638,
    0.47178664,
    0.51769992,
    0.57647527,
    0.65007685,
    0.74384196,
    0.83274599,
    0.91602345,
    1.0,
]
svm_spec_tpr = [
    0,
    0.05212551,
    0.10425101,
    0.15819838,
    0.21661606,
    0.28469973,
    0.34642375,
    0.39848178,
    0.44908907,
    0.50214238,
    0.55624157,
    0.60747301,
    0.65787787,
    0.71224696,
    0.7641363,
    0.81368084,
    0.85936235,
    0.90237854,
    0.9417004,
    0.98076923,
]
svm_spec_auc = 0.606
# rfc spec
rfc_spec_fpr = [
    0,
    0.00153945,
    0.0030789,
    0.00461835,
    0.00615779,
    0.00769724,
    0.00923669,
    0.0108333,
    0.01245848,
    0.01408366,
    0.05495637,
    0.13507659,
    0.21519681,
    0.30516433,
    0.41482644,
    0.53030903,
    0.64773177,
    0.76515451,
    0.88257726,
    1.0,
]
rfc_spec_tpr = [
    0,
    0.02559042,
    0.05118084,
    0.07677126,
    0.10236167,
    0.12795209,
    0.15354251,
    0.1775135,
    0.20067476,
    0.22383603,
    0.2652834,
    0.32501687,
    0.38475034,
    0.45182186,
    0.5335695,
    0.62139001,
    0.71123482,
    0.80107962,
    0.89092443,
    0.98076923,
]
rfc_spec_auc = 0.598

plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(rfc_spec_fpr, rfc_spec_tpr, marker=".", label="RFC AUC=" + str(rfc_spec_auc))
plt.plot(svm_spec_fpr, svm_spec_tpr, marker=".", label="SVM AUC=" + str(svm_spec_auc))
plt.plot(knn_spec_fpr, knn_spec_tpr, marker=".", label="KNN AUC=" + str(knn_spec_auc))
plt.plot(lda_spec_fpr, lda_spec_tpr, marker=".", label="LDA AUC=" + str(lda_spec_auc))

# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(label="ROC curve - Spectral features", fontsize=20)

# show the legend
plt.legend()
# show the plot
plt.savefig(
    os.path.join(cfg.fig_dir, "roc_curve_spec " + cfg.result_name_file + ".png")
)


# lda stat
lda_stat_fpr = [
    0,
    0.00168745,
    0.00450126,
    0.01191045,
    0.01824168,
    0.02492525,
    0.03398941,
    0.04111189,
    0.05054671,
    0.06854919,
    0.08479823,
    0.09614889,
    0.11244636,
    0.13500943,
    0.17071363,
    0.21686266,
    0.25641102,
    0.34854167,
    0.65353554,
    1.0,
]
lda_stat_tpr = [
    0,
    0.02419028,
    0.04919028,
    0.08753374,
    0.15059042,
    0.22687247,
    0.28785425,
    0.35708502,
    0.42307692,
    0.47945344,
    0.53696019,
    0.59971323,
    0.65381242,
    0.70754049,
    0.75779352,
    0.80522942,
    0.88549258,
    0.96766194,
    0.98076923,
    0.98076923,
]
lda_stat_auc = 0.877
# knn stat
knn_stat_fpr = [
    0,
    0.0010008,
    0.00224486,
    0.00683112,
    0.01514495,
    0.02306569,
    0.02950563,
    0.03556762,
    0.04027901,
    0.04877287,
    0.0578041,
    0.06632051,
    0.07237032,
    0.08337862,
    0.10095464,
    0.12863172,
    0.17785641,
    0.34473043,
    0.66351606,
    1.0,
]
knn_stat_tpr = [
    0,
    0.04159919,
    0.08218623,
    0.11821862,
    0.16255061,
    0.21791498,
    0.27584345,
    0.33103914,
    0.40153509,
    0.44617072,
    0.48211876,
    0.52680499,
    0.59471997,
    0.65482456,
    0.7082996,
    0.74564777,
    0.76661606,
    0.81798246,
    0.89897099,
    0.98076923,
]
knn_stat_auc = 0.822
# svm stat
svm_stat_fpr = [
    0,
    0.00153436,
    0.01206055,
    0.05351255,
    0.08108232,
    0.099299,
    0.13344563,
    0.16175389,
    0.18819047,
    0.22258141,
    0.25026973,
    0.27132223,
    0.29423671,
    0.32865556,
    0.37592098,
    0.44872138,
    0.53037942,
    0.60215194,
    0.78234269,
    1.0,
]
svm_stat_tpr = [
    0,
    0.03549258,
    0.07442645,
    0.11088057,
    0.17285762,
    0.24477058,
    0.3222166,
    0.40752362,
    0.46519906,
    0.50512821,
    0.55931174,
    0.62754723,
    0.68314777,
    0.72712551,
    0.7794197,
    0.82810391,
    0.88166329,
    0.94698043,
    0.96988866,
    0.98076923,
]
svm_stat_auc = 0.741
# rfc stat
rfc_stat_fpr = [
    0,
    0.00180965,
    0.0036193,
    0.00542895,
    0.00723859,
    0.00904824,
    0.01085789,
    0.01266754,
    0.01447719,
    0.01628684,
    0.06891842,
    0.17237192,
    0.27582543,
    0.37927894,
    0.48273245,
    0.58618596,
    0.68963947,
    0.79309298,
    0.89654649,
    1.0,
]
rfc_stat_tpr = [
    0,
    0.01754386,
    0.03508772,
    0.05263158,
    0.07017544,
    0.0877193,
    0.10526316,
    0.12280702,
    0.14035088,
    0.15789474,
    0.20951417,
    0.29520918,
    0.38090418,
    0.46659919,
    0.5522942,
    0.6379892,
    0.72368421,
    0.80937922,
    0.89507422,
    0.98076923,
]
rfc_stat_auc = 0.565

plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(rfc_stat_fpr, rfc_stat_tpr, marker=".", label="RFC AUC=" + str(rfc_stat_auc))
plt.plot(svm_stat_fpr, svm_stat_tpr, marker=".", label="SVM AUC=" + str(svm_stat_auc))
plt.plot(knn_stat_fpr, knn_stat_tpr, marker=".", label="KNN AUC=" + str(knn_stat_auc))
plt.plot(lda_stat_fpr, lda_stat_tpr, marker=".", label="LDA AUC=" + str(lda_stat_auc))

# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(label="ROC curve - Statistical features", fontsize=20)

# show the legend
plt.legend()
# show the plot
plt.savefig(
    os.path.join(cfg.fig_dir, "roc_curve_stat " + cfg.result_name_file + ".png")
)


# rfc vgg
rfc_vgg_fpr = [
    0,
    0.00598099,
    0.01196198,
    0.01794297,
    0.02390484,
    0.03020092,
    0.03772894,
    0.04895779,
    0.06226542,
    0.07591559,
    0.11827434,
    0.18934165,
    0.26570049,
    0.3576837,
    0.47386027,
    0.58398763,
    0.68863933,
    0.79242622,
    0.89621311,
    1.0,
]
rfc_vgg_tpr = [
    0,
    0.01811741,
    0.03623482,
    0.05435223,
    0.07246964,
    0.0909919,
    0.11072874,
    0.13579622,
    0.16393387,
    0.19267881,
    0.2425776,
    0.31363023,
    0.39030027,
    0.47510121,
    0.56867409,
    0.65738866,
    0.73922065,
    0.81973684,
    0.90025304,
    0.98076923,
]
rfc_vgg_auc = 0.573
# svm vgg
# lda vgg
lda_vgg_fpr = [
    0,
    0.03592032,
    0.0752891,
    0.11292031,
    0.15355187,
    0.18963998,
    0.21985548,
    0.24923004,
    0.27706626,
    0.30439839,
    0.33281044,
    0.36172837,
    0.3899331,
    0.42238682,
    0.45273514,
    0.48800186,
    0.52723383,
    0.58112952,
    0.65637031,
    1.0,
]
lda_vgg_tpr = [
    0,
    0.20291835,
    0.32520243,
    0.42928475,
    0.50538124,
    0.55907557,
    0.60850202,
    0.63773617,
    0.67184548,
    0.69220648,
    0.7138664,
    0.73918691,
    0.76894399,
    0.78898448,
    0.80990216,
    0.84256073,
    0.86361336,
    0.87786775,
    0.896778,
    0.98076923,
]
lda_vgg_auc = 0.750
# knn vgg
knn_vgg_fpr = [
    0,
    0.00144011,
    0.00296621,
    0.00617714,
    0.01189625,
    0.01830049,
    0.02280227,
    0.02957801,
    0.03827227,
    0.0485101,
    0.0602177,
    0.07244192,
    0.08564049,
    0.11810758,
    0.18434892,
    0.25687381,
    0.37311518,
    0.55828906,
    0.77771121,
    1.0,
]
knn_vgg_tpr = [
    0,
    0.00759109,
    0.01518219,
    0.02226721,
    0.03233806,
    0.04600202,
    0.07211538,
    0.10096154,
    0.13415992,
    0.16204453,
    0.18896761,
    0.22054656,
    0.25551619,
    0.29787449,
    0.36740891,
    0.4340081,
    0.53643725,
    0.67449393,
    0.82732794,
    0.98076923,
]
knn_vgg_auc = 0.606
plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(rfc_vgg_fpr, rfc_vgg_tpr, marker=".", label="RFC AUC=" + str(rfc_vgg_auc))
# plt.plot(svm_vgg_fpr, svm_vgg_tpr, marker=".", label="SVM AUC=" + str(svm_vgg_auc))
plt.plot(knn_vgg_fpr, knn_vgg_tpr, marker=".", label="KNN AUC=" + str(knn_vgg_auc))
plt.plot(lda_vgg_fpr, lda_vgg_tpr, marker=".", label="LDA AUC=" + str(lda_vgg_auc))

# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(label="ROC curve - VGG16 features", fontsize=20)
# show the legend
plt.legend()
# show the plot
plt.savefig(os.path.join(cfg.fig_dir, "roc_curve_vgg " + cfg.result_name_file + ".png"))


# knn Mobnet
knn_Mobnet_fpr = [
    0.00000000e00,
    9.43442069e-04,
    1.88688414e-03,
    3.12179074e-03,
    5.82696615e-03,
    1.09168101e-02,
    1.55427493e-02,
    2.37026863e-02,
    3.22472244e-02,
    3.94334129e-02,
    4.76545447e-02,
    5.72155119e-02,
    6.84668046e-02,
    9.82772258e-02,
    1.64017415e-01,
    2.36527690e-01,
    3.59330153e-01,
    5.45959950e-01,
    7.72979975e-01,
    1.00000000e00,
]
knn_Mobnet_tpr = [
    0,
    0.01155831,
    0.02311662,
    0.0376161,
    0.05361197,
    0.06940144,
    0.08682491,
    0.10404197,
    0.13278294,
    0.16730306,
    0.19905401,
    0.22686619,
    0.2495184,
    0.29186447,
    0.37146543,
    0.44406605,
    0.55146199,
    0.69057448,
    0.84528724,
    1.0,
]
knn_Mobnet_auc = 0.630
# svm Mobnet
svm_Mobnet_fpr = [
    0,
    0.00180064,
    0.00924672,
    0.03481458,
    0.05119998,
    0.06024843,
    0.08303855,
    0.12275734,
    0.15021552,
    0.17218244,
    0.20843733,
    0.25745852,
    0.29038333,
    0.31064858,
    0.3730715,
    0.48591825,
    0.56550755,
    0.62617341,
    0.80552696,
    1.0,
]
svm_Mobnet_tpr = [
    0,
    0.02650499,
    0.05197798,
    0.07724458,
    0.1378053,
    0.2378053,
    0.29802202,
    0.33558652,
    0.40416237,
    0.48114895,
    0.5378913,
    0.58197454,
    0.64927761,
    0.73887169,
    0.79193326,
    0.81584107,
    0.89143447,
    0.98570691,
    1,
    1.0,
]
svm_Mobnet_auc = 0.760
# lda Mobnet
lda_Mobnet_fpr = [
    0,
    0.00282491,
    0.01778944,
    0.04005895,
    0.05256231,
    0.06974364,
    0.09512002,
    0.12572384,
    0.1556385,
    0.18105475,
    0.21241677,
    0.25236183,
    0.30311916,
    0.34803614,
    0.40317939,
    0.49482018,
    0.57100894,
    0.60888759,
    0.76626308,
    1.0,
]
lda_Mobnet_tpr = [
    0,
    0.03304656,
    0.06584008,
    0.10738866,
    0.19649123,
    0.28473347,
    0.34596829,
    0.40053981,
    0.46256748,
    0.52149123,
    0.58692645,
    0.65106275,
    0.70777665,
    0.76997301,
    0.81729082,
    0.84768893,
    0.88776991,
    0.95747301,
    0.98076923,
    0.98076923,
]
lda_Mobnet_auc = 0.766
# tree Mobnet
rfc_Mobnet_fpr = [
    0,
    0.00266462,
    0.00532924,
    0.00799386,
    0.01056808,
    0.01332313,
    0.01689226,
    0.03313733,
    0.05579628,
    0.07856911,
    0.12939542,
    0.20827521,
    0.29036866,
    0.38202048,
    0.48850416,
    0.59030748,
    0.6900894,
    0.79339293,
    0.89669647,
    1.0,
]
rfc_Mobnet_tpr = [
    0,
    0.01940144,
    0.03880289,
    0.05820433,
    0.07610939,
    0.09355005,
    0.1128483,
    0.14425525,
    0.18171655,
    0.21917785,
    0.27433781,
    0.34719642,
    0.42377021,
    0.50361197,
    0.58503612,
    0.66800826,
    0.7495356,
    0.83302374,
    0.91651187,
    1.0,
]
rfc_Mobnet_auc = 0.586


plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(
    rfc_Mobnet_fpr, rfc_Mobnet_tpr, marker=".", label="RFC AUC=" + str(rfc_Mobnet_auc)
)
plt.plot(
    svm_Mobnet_fpr, svm_Mobnet_tpr, marker=".", label="SVM AUC=" + str(svm_Mobnet_auc)
)
plt.plot(
    knn_Mobnet_fpr, knn_Mobnet_tpr, marker=".", label="KNN AUC=" + str(knn_Mobnet_auc)
)
plt.plot(
    lda_Mobnet_fpr, lda_Mobnet_tpr, marker=".", label="LDA AUC=" + str(lda_Mobnet_auc)
)

# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(label="ROC curve - MobileNet features", fontsize=20)
# show the legend
plt.legend()
# show the plot
plt.savefig(
    os.path.join(cfg.fig_dir, "roc_curve_MobileNet " + cfg.result_name_file + ".png")
)


# FCN
FCN_fpr = [
    0,
    0.00687174,
    0.0377328,
    0.10642922,
    0.16556885,
    0.25095806,
    0.344492,
    0.5146729,
    0.63047026,
    1,
]
FCN_tpr = [
    0,
    0.0962963,
    0.27770516,
    0.43562092,
    0.57207698,
    0.68024691,
    0.79324619,
    0.86768337,
    0.98409586,
    1,
]
FCN_auc = 0.798
plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(FCN_fpr, FCN_tpr, marker=".", label="FCN AUC=" + str(FCN_auc))


# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(label="ROC curve - FCN model", fontsize=20)
# show the legend
plt.legend()
# show the plot
plt.savefig(os.path.join(cfg.fig_dir, "roc_curve_FCN " + cfg.result_name_file + ".png"))


classifiers = ["lda", "knn", "svm", "rfc"]
attrs = ["all", "spec", "stat", "temp", "vgg", "Mobnet"]
eer = list()
framwork = list()
auc = list()
for classifier in classifiers:
    for attr in attrs:
        print(type(attr))
        if attr == "vgg" and classifier == "svm":
            continue
        A = np.array(eval(classifier + "_" + attr + "_tpr"))
        B = np.array(eval(classifier + "_" + attr + "_fpr"))
        eer.append(round(compute_eer(B, A), 3))
        auc.append(round(eval(classifier + "_" + attr + "_auc"), 3))
        framwork.append((classifier + "_" + attr).upper())

A = np.array(FCN_tpr)
B = np.array(FCN_fpr)
eer.append(round(compute_eer(B, A), 3))
framwork.append("FCN")
auc.append(round(FCN_auc, 3))


print(framwork)
print(eer)
print(auc)
print(len(framwork))
print(len(auc))
print(len(eer))
import pandas as pd
import os

tab = pd.DataFrame([framwork, auc, eer], index=["Frame Work", "AUC", "EER"])
print(tab.T.head())
with open(os.path.join(cfg.tbl_dir, "auc.tex"), "w") as tf:
    tf.write(tab.T.to_latex())
# plt.show()

