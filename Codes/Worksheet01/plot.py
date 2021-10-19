import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path as Pathlb


from scipy.spatial import distance

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from itertools import combinations, product
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf


plt.rcParams["font.size"] = 13
plt.tight_layout()

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
data_dir = os.path.join(project_dir, "Archive", "results_All")
data_dir = os.path.join(project_dir, "results COXTS")

Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)



Results_DF = pd.read_excel(os.path.join(data_dir, 'Results_DF.xlsx'), index_col = 0)

print(Results_DF)

test_ratios = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
persentages = [1.0]
Modes = ["corr", "dist"]
model_types = ["min", "median", "average"]
THRESHOLDs = perf.THRESHOLDs
normilizings = ["z-score"]#, "minmax", "None"]
feature_names = perf.feature_names

color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink', 'blueviolet']









Results_DF_temp = Results_DF[   Results_DF["Features_Set"] != "All"   ]
Results_DF_temp = Results_DF_temp[   Results_DF_temp["Mean_EER_Left"] != 0   ]
Results_DF_temp.columns = perf.cols


# X = Results_DF_temp.sort_values(by=['Acc_Left', 'EER_Left'], ascending = [False, True]).iloc[:10,:8]
# with open(os.path.join("Manuscripts", "src", "tables", "top10_left.tex"), "w") as tf:
#     tf.write(X.round(decimals=2).to_latex())

# X = Results_DF_temp.sort_values(by=['Acc_Left', 'EER_Left'], ascending = [True, False]).iloc[:10,:8]
# with open(os.path.join("Manuscripts", "src", "tables", "worse10_left.tex"), "w") as tf:
#     tf.write(X.round(decimals=2).to_latex())






# X = Results_DF_temp.sort_values(by=['Acc_Right', 'EER_Right'], ascending = [False, True]).iloc[:10,:10].drop(columns =['Acc_Left', 'EER_Left'])
# with open(os.path.join("Manuscripts", "src", "tables", "top10_right.tex"), "w") as tf:
#     tf.write(X.round(decimals=2).to_latex())      


# X = Results_DF_temp.sort_values(by=['Acc_Right', 'EER_Right'], ascending = [True, False]).iloc[:10,:10].drop(columns =['Acc_Left', 'EER_Left'])
# with open(os.path.join("Manuscripts", "src", "tables", "worse10_right.tex"), "w") as tf:
#     tf.write(X.round(decimals=2).to_latex())          



Results_DF_all = Results_DF[   Results_DF["Features_Set"] == "All"   ]





auc=[]
plt.figure(figsize=(14,8))
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
data = list()
X = pd.DataFrame(index=["Correlation", "Euclidean distance"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["Correlation", "Euclidean distance"] , columns=[ "EER Left", "EER Right"])
for idx, temp in enumerate(Modes):
    a = ["Correlation", "Euclidean distance"]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Mode"] == temp   ]

    data.append(Results_DF_all_mode["Mean_Accuracy_Left"].values)


    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Left"].mean(),  Results_DF_all_mode["Mean_Accuracy_Left"].std(), Results_DF_all_mode["Mean_Accuracy_Left"].min(), Results_DF_all_mode["Mean_Accuracy_Left"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Right"].mean(), Results_DF_all_mode["Mean_Accuracy_Right"].std(), Results_DF_all_mode["Mean_Accuracy_Right"].min(), Results_DF_all_mode["Mean_Accuracy_Right"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Left"].mean(),       Results_DF_all_mode["Mean_EER_Left"].std(), Results_DF_all_mode["Mean_EER_Left"].min(), Results_DF_all_mode["Mean_EER_Left"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Right"].mean(),      Results_DF_all_mode["Mean_EER_Right"].std(), Results_DF_all_mode["Mean_EER_Right"].min(), Results_DF_all_mode["Mean_EER_Right"].max())    
       
    # plt.boxplot(Results_DF_all_mode["Mean_Accuracy_Left"])
    # plt.show()

    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))
    label=a[idx] #+ ' AUC = ' + str(round(auc, 2))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, left side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

    plt.subplot(1,2,2)
    auc = (1 + np.trapz( FRR_R[idx], FAR_R[idx]))
    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

PATH = os.path.join("Manuscripts", "src", "figures", "Correlation.png")
plt.tight_layout()
plt.savefig(PATH)
plt.close('all')

stat, p = shapiro(data[1])
if p > 0.05:
    print('[INFO] Probably Gaussian,\t\t\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    stat, p = ttest_ind(data[0], data[1])
    if p > 0.05:
        print('[INFO] Probably the same distribution,\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    else:
        print('[INFO] Probably different distributions,\t\t stat=%.3f,\t p=%.3f' % (stat, p))
else:
    print('[INFO] Probably not Gaussian,\t\t\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    stat, p = mannwhitneyu(data[0], data[1])
    if p > 0.05:
        print('[INFO] Probably the same distribution,\t\t stat=%.3f,\t p=%.3f' % (stat, p))
    else:
        print('[INFO] Probably different distributions,\t\t stat=%.3f,\t p=%.3f' % (stat, p))

with open(os.path.join("Manuscripts", "src", "tables", "Correlation.tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "Correlation1.tex"), "w") as tf:
        tf.write(Y.round(decimals=2).to_latex())
#########################################################################################################
#########################################################################################################
#########################################################################################################
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["Minimum", "Median", "Average"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["Minimum", "Median", "Average"] , columns=[ "EER Left", "EER Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
data = list()
for idx, temp in enumerate(model_types):
    a = ["Minimum", "Median", "Average"]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Model_Type"] == temp   ]
    data.append(Results_DF_all_mode["Mean_Accuracy_Left"].values)

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Left"].mean(),  Results_DF_all_mode["Mean_Accuracy_Left"].std(), Results_DF_all_mode["Mean_Accuracy_Left"].min(), Results_DF_all_mode["Mean_Accuracy_Left"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Right"].mean(), Results_DF_all_mode["Mean_Accuracy_Right"].std(), Results_DF_all_mode["Mean_Accuracy_Right"].min(), Results_DF_all_mode["Mean_Accuracy_Right"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Left"].mean(),       Results_DF_all_mode["Mean_EER_Left"].std(), Results_DF_all_mode["Mean_EER_Left"].min(), Results_DF_all_mode["Mean_EER_Left"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Right"].mean(),      Results_DF_all_mode["Mean_EER_Right"].std(), Results_DF_all_mode["Mean_EER_Right"].min(), Results_DF_all_mode["Mean_EER_Right"].max())    

    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)




    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))
    plt.subplot(1,2,1)
    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)
    
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, left side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")


    plt.subplot(1,2,2)
    auc = (1 + np.trapz( FRR_R[idx], FAR_R[idx]))

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")
   




PATH = os.path.join("Manuscripts", "src", "figures", "Minimum.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "Minimum.tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "Minimum1.tex"), "w") as tf:
        tf.write(Y.round(decimals=2).to_latex())        
#########################################################################################################
#########################################################################################################
#########################################################################################################
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["Z-score algorithm", "MinMax algorithm", "None"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["Z-score algorithm", "MinMax algorithm", "None"] , columns=[ "EER Left", "EER Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()   
for idx, temp in enumerate(normilizings):
    a = ["Z-score algorithm", "MinMax algorithm", "None"]


    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Normalizition"] == temp   ]

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Left"].mean(),  Results_DF_all_mode["Mean_Accuracy_Left"].std(), Results_DF_all_mode["Mean_Accuracy_Left"].min(), Results_DF_all_mode["Mean_Accuracy_Left"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Right"].mean(), Results_DF_all_mode["Mean_Accuracy_Right"].std(), Results_DF_all_mode["Mean_Accuracy_Right"].min(), Results_DF_all_mode["Mean_Accuracy_Right"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Left"].mean(),       Results_DF_all_mode["Mean_EER_Left"].std(), Results_DF_all_mode["Mean_EER_Left"].min(), Results_DF_all_mode["Mean_EER_Left"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Right"].mean(),      Results_DF_all_mode["Mean_EER_Right"].std(), Results_DF_all_mode["Mean_EER_Right"].min(), Results_DF_all_mode["Mean_EER_Right"].max())    

    
    
    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)



    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, left side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

    plt.subplot(1,2,2)
    auc = (1 + np.trapz( FRR_R[idx], FAR_R[idx]))

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

PATH = os.path.join("Manuscripts", "src", "figures", "MinMax.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "MinMax.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "MinMax1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())
#########################################################################################################
#########################################################################################################
#########################################################################################################
a = ["10 percent", "20 percent", "25 percent", "30 percent", "40 percent", "50 percent", "60 percent","70 percent", "75 percent", "80 percent" , "90 percent"]

plt.figure(figsize=(14,8))
X = pd.DataFrame(index=a , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=a , columns=[ "EER Left", "EER Right"])
Z = pd.DataFrame(index=a , columns=[ "Left", "Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(test_ratios):
    a = ["10 percent", "20 percent", "25 percent", "30 percent", "40 percent", "50 percent", "60 percent","70 percent", "75 percent", "80 percent" , "90 percent"]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Test_Size"] == temp   ].reset_index()

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Left"].mean(),  Results_DF_all_mode["Mean_Accuracy_Left"].std(), Results_DF_all_mode["Mean_Accuracy_Left"].min(), Results_DF_all_mode["Mean_Accuracy_Left"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Right"].mean(), Results_DF_all_mode["Mean_Accuracy_Right"].std(), Results_DF_all_mode["Mean_Accuracy_Right"].min(), Results_DF_all_mode["Mean_Accuracy_Right"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Left"].mean(),       Results_DF_all_mode["Mean_EER_Left"].std(), Results_DF_all_mode["Mean_EER_Left"].min(), Results_DF_all_mode["Mean_EER_Left"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Right"].mean(),      Results_DF_all_mode["Mean_EER_Right"].std(), Results_DF_all_mode["Mean_EER_Right"].min(), Results_DF_all_mode["Mean_EER_Right"].max())    

    print(        Results_DF_all_mode.at[0, "Mean_sample_test_Left"])
    Z.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:2.2f}, {:2.2f})".format(
        Results_DF_all_mode.at[0, "Mean_sample_test_Left"], 
        Results_DF_all_mode.at[0, "std_sample_test_Left"],
        Results_DF_all_mode.at[0, "Min_sample_test_Left"],
        Results_DF_all_mode.at[0, "Max_sample_test_Left"])

    Z.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:2.2f}, {:2.2f})".format(
        Results_DF_all_mode.at[0, "Mean_sample_test_Right"], 
        Results_DF_all_mode.at[0, "std_sample_test_Right"],
        Results_DF_all_mode.at[0, "Min_sample_test_Right"],
        Results_DF_all_mode.at[0, "Max_sample_test_Right"])



    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)




    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, left side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

    plt.subplot(1,2,2)
    auc = (1 + np.trapz( FRR_R[idx], FAR_R[idx]))

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")


PATH = os.path.join("Manuscripts", "src", "figures",  "testsize.png")
plt.tight_layout()
plt.savefig(PATH)
plt.close('all')
with open(os.path.join("Manuscripts", "src", "tables", "testsize.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "testsize1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "testsize2.tex"), "w") as tf:
    tf.write(Z.round(decimals=2).to_latex())

#########################################################################################################
#########################################################################################################
#########################################################################################################
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
plt.figure(figsize=(14,8))
X = pd.DataFrame(index=["All PCs", "Keeping 95 percent of variance"] , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=["All PCs", "Keeping 95 percent of variance"] , columns=[ "EER Left", "EER Right"])

for idx, temp in enumerate(persentages):
    a = ["All PCs", "Keeping 95 percent of variance"]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["PCA"] == temp   ]

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Left"].mean(),  Results_DF_all_mode["Mean_Accuracy_Left"].std(), Results_DF_all_mode["Mean_Accuracy_Left"].min(), Results_DF_all_mode["Mean_Accuracy_Left"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_Accuracy_Right"].mean(), Results_DF_all_mode["Mean_Accuracy_Right"].std(), Results_DF_all_mode["Mean_Accuracy_Right"].min(), Results_DF_all_mode["Mean_Accuracy_Right"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Left"].mean(),       Results_DF_all_mode["Mean_EER_Left"].std(), Results_DF_all_mode["Mean_EER_Left"].min(), Results_DF_all_mode["Mean_EER_Left"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_all_mode["Mean_EER_Right"].mean(),      Results_DF_all_mode["Mean_EER_Right"].std(), Results_DF_all_mode["Mean_EER_Right"].min(), Results_DF_all_mode["Mean_EER_Right"].max())    



    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)


    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, left side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

    plt.subplot(1,2,2)

    auc = (1 + np.trapz( FRR_R[idx], FAR_R[idx]))

    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")


PATH = os.path.join("Manuscripts", "src", "figures",  "PCA.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "PCA.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "PCA1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())





#########################################################################################################
#########################################################################################################
#########################################################################################################
a = ["All features", "Only MDIST features", "Only RDIST features", "Only TOTEX features", "Only MVELO features", "Only RANGE features", "Only AREAXX features", "Only MFREQ features", "Only FDPD features", "Only FDCX features"]    

plt.figure(figsize=(14,8))
X = pd.DataFrame(index=a , columns=["Accuracy Left", "Accuracy Right"])
Y = pd.DataFrame(index=a , columns=[ "EER Left", "EER Right"])

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(feature_names):
    a = ["All features", "Only MDIST features", "Only RDIST features", "Only TOTEX features", "Only MVELO features", "Only RANGE features", "Only AREAXX features", "Only MFREQ features", "Only FDPD features", "Only FDCX features"]    
    Results_DF_temp = Results_DF[   Results_DF["Features_Set"] == temp   ]
    Results_DF_temp = Results_DF_temp[   Results_DF_temp["PCA"] == 1   ]

    X.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_Accuracy_Left"].mean(),  Results_DF_temp["Mean_Accuracy_Left"].std(), Results_DF_temp["Mean_Accuracy_Left"].min(), Results_DF_temp["Mean_Accuracy_Left"].max())
    X.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_Accuracy_Right"].mean(), Results_DF_temp["Mean_Accuracy_Right"].std(), Results_DF_temp["Mean_Accuracy_Right"].min(), Results_DF_temp["Mean_Accuracy_Right"].max())
    Y.iloc[idx,0] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_EER_Left"].mean(),       Results_DF_temp["Mean_EER_Left"].std(), Results_DF_temp["Mean_EER_Left"].min(), Results_DF_temp["Mean_EER_Left"].max())
    Y.iloc[idx,1] = "{:2.2f} +/- {:2.2f} ({:.2f}, {:.2f})".format(Results_DF_temp["Mean_EER_Right"].mean(),      Results_DF_temp["Mean_EER_Right"].std(), Results_DF_temp["Mean_EER_Right"].min(), Results_DF_temp["Mean_EER_Right"].max())    

    cols = ["FAR_L_" + str(i) for i in range(100)] 
    FAR_L.append(Results_DF_temp.loc[:, cols].mean().values)

    cols = ["FRR_L_" + str(i) for i in range(100)]
    FRR_L.append(Results_DF_temp.loc[:, cols].mean().values)

    cols = ["FAR_R_" + str(i) for i in range(100)] 
    FAR_R.append(Results_DF_temp.loc[:, cols].mean().values)

    cols = ["FRR_R_" + str(i) for i in range(100)]
    FRR_R.append(Results_DF_temp.loc[:, cols].mean().values)


    plt.subplot(1,2,1)
    auc = (1 + np.trapz( FRR_L[idx], FAR_L[idx]))

    plt.plot(FAR_L[idx], FRR_L[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)

    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, left side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

    plt.subplot(1,2,2)
    auc = (1 + np.trapz( FRR_R[idx], FAR_R[idx]))
    label = a[idx] #+ ' AUC = ' + str(round(auc, 2))
    plt.plot(FAR_R[idx], FRR_R[idx], linestyle='--', marker='o', color=color[idx], lw = 2, label=a[idx], clip_on=False)



    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('False Rejection Rate')
    plt.title('ROC curve, Right side')
    plt.gca().set_aspect('equal')
    plt.legend(loc="best")

PATH = os.path.join("Manuscripts", "src", "figures", "feat.png")
plt.tight_layout()
plt.savefig(PATH )
plt.close('all')

with open(os.path.join("Manuscripts", "src", "tables", "feat.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())
with open(os.path.join("Manuscripts", "src", "tables", "feat1.tex"), "w") as tf:
    tf.write(Y.round(decimals=2).to_latex())