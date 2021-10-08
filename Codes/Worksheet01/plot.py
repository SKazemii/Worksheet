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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf


plt.rcParams["font.size"] = 13
plt.tight_layout()

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
data_dir = os.path.join(project_dir, "results_All")

Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)


if False:
    # if True:
    Results_DF = pd.read_excel(os.path.join(data_dir, 'Results_DF.xlsx'), index_col = 0)

    Results_DF1 = pd.read_excel(os.path.join(data_dir, 'Results_DF1.xlsx'), index_col = 0)

    Results_DF = pd.concat([Results_DF.reset_index(), Results_DF1], axis=0)

    Results_DF.to_excel(os.path.join(data_dir, 'Results_DF_all.xlsx'))
    
else:
    # Results_DF = pd.read_excel(os.path.join(data_dir, "excels", 'Results_DF_all.xlsx'), index_col = 0)
    Results_DF = pd.read_excel(os.path.join(data_dir, 'Results_DF_all.xlsx'), index_col = 0)
# sys.exit()


test_ratios = [0.2, 0.35, 0.5]
persentages = [1.0, 0.95]
Modes = ["corr", "dist"]
model_types = ["min", "median", "average"]
THRESHOLDs = np.linspace(0, 1, 100)
normilizings = ["z-score", "minmax", "None"]
feature_names = ["All", "MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]

color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink']




Results_DF_temp = Results_DF[   Results_DF["Features_Set"] != "All"   ]
Results_DF_temp = Results_DF_temp[   Results_DF_temp["Mean_EER_Left"] != 0   ]
Results_DF_temp.columns = ["Mode", "Model_Type", "TestSize", "Norm", "Features_Set", "PCA", "Acc_Left", "EER_Left", "Acc_Right", "EER_Right","Min_Accuracy_Left", "Min_EER_Left", "Min_Accuracy_Right", "Min_EER_Right",
"Max_Accuracy_Left", "Max_EER_Left", "Max_Accuracy_Right", "Max_EER_Right",
"Median_Accuracy_Left", "Median_EER_Left", "Median_Accuracy_Right", "Median_EER_Right"] + ["FAR_L_" + str(i) for i in range(100)] + ["FRR_L_" + str(i) for i in range(100)] + ["FAR_R_" + str(i) for i in range(100)] + ["FRR_R_" + str(i) for i in range(100)]


X = Results_DF_temp.sort_values(by=['Acc_Left', 'EER_Left'], ascending = [False, True]).iloc[:10,:8]
with open(os.path.join("Manuscripts", "src", "tables", "top10_left.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())

X = Results_DF_temp.sort_values(by=['Acc_Left', 'EER_Left'], ascending = [True, False]).iloc[:10,:8]
with open(os.path.join("Manuscripts", "src", "tables", "worse10_left.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())






X = Results_DF_temp.sort_values(by=['Acc_Right', 'EER_Right'], ascending = [False, True]).iloc[:10,:10].drop(columns =['Acc_Left', 'EER_Left'])
with open(os.path.join("Manuscripts", "src", "tables", "top10_right.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())      


X = Results_DF_temp.sort_values(by=['Acc_Right', 'EER_Right'], ascending = [True, False]).iloc[:10,:10].drop(columns =['Acc_Left', 'EER_Left'])
with open(os.path.join("Manuscripts", "src", "tables", "worse10_right.tex"), "w") as tf:
    tf.write(X.round(decimals=2).to_latex())          



Results_DF_all = Results_DF[   Results_DF["Features_Set"] == "All"   ]





auc=[]
plt.figure(figsize=(14,8))
npy = np.empty((4,4))
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(Modes):
    a = ["Correlation", "Euclidean Distance"]
    

    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Mode"] == temp   ]

    npy[0,0] = Results_DF_all_mode["Mean_Accuracy_Left"].mean()  
    npy[0,1] = Results_DF_all_mode["Mean_Accuracy_Right"].mean()  
    npy[0,2] = Results_DF_all_mode["Mean_EER_Left"].mean()  
    npy[0,3] = Results_DF_all_mode["Mean_EER_Right"].mean()

    npy[1,0] = Results_DF_all_mode["Mean_Accuracy_Left"].min()  
    npy[1,1] = Results_DF_all_mode["Mean_Accuracy_Right"].min()  
    npy[1,2] = Results_DF_all_mode["Mean_EER_Left"].min()  
    npy[1,3] = Results_DF_all_mode["Mean_EER_Right"].min()  

    npy[2,0] = Results_DF_all_mode["Mean_Accuracy_Left"].max()  
    npy[2,1] = Results_DF_all_mode["Mean_Accuracy_Right"].max()  
    npy[2,2] = Results_DF_all_mode["Mean_EER_Left"].max()  
    npy[2,3] = Results_DF_all_mode["Mean_EER_Right"].max()    

    npy[3,0] = Results_DF_all_mode["Mean_Accuracy_Left"].median()  
    npy[3,1] = Results_DF_all_mode["Mean_Accuracy_Right"].median()  
    npy[3,2] = Results_DF_all_mode["Mean_EER_Left"].median()  
    npy[3,3] = Results_DF_all_mode["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])
    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())

    
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

PATH = os.path.join("Manuscripts", "src", "figures", "Correlation")
plt.tight_layout()
plt.savefig(PATH + "_ROC.png")
plt.close('all')


#########################################################################################################
#########################################################################################################
#########################################################################################################
plt.figure(figsize=(14,8))

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(model_types):
    a = ["Minimum", "Median", "Average"]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Model_Type"] == temp   ]

   
    npy[0,0] = Results_DF_all_mode["Mean_Accuracy_Left"].mean()  
    npy[0,1] = Results_DF_all_mode["Mean_Accuracy_Right"].mean()  
    npy[0,2] = Results_DF_all_mode["Mean_EER_Left"].mean()  
    npy[0,3] = Results_DF_all_mode["Mean_EER_Right"].mean()

    npy[1,0] = Results_DF_all_mode["Mean_Accuracy_Left"].min()  
    npy[1,1] = Results_DF_all_mode["Mean_Accuracy_Right"].min()  
    npy[1,2] = Results_DF_all_mode["Mean_EER_Left"].min()  
    npy[1,3] = Results_DF_all_mode["Mean_EER_Right"].min()  

    npy[2,0] = Results_DF_all_mode["Mean_Accuracy_Left"].max()  
    npy[2,1] = Results_DF_all_mode["Mean_Accuracy_Right"].max()  
    npy[2,2] = Results_DF_all_mode["Mean_EER_Left"].max()  
    npy[2,3] = Results_DF_all_mode["Mean_EER_Right"].max()    

    npy[3,0] = Results_DF_all_mode["Mean_Accuracy_Left"].median()  
    npy[3,1] = Results_DF_all_mode["Mean_Accuracy_Right"].median()  
    npy[3,2] = Results_DF_all_mode["Mean_EER_Left"].median()  
    npy[3,3] = Results_DF_all_mode["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])

    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())

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
   




PATH = os.path.join("Manuscripts", "src", "figures", "Median")
plt.tight_layout()
plt.savefig(PATH + "_ROC.png")
plt.close('all')

#########################################################################################################
#########################################################################################################
#########################################################################################################
plt.figure(figsize=(14,8))

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()   
for idx, temp in enumerate(normilizings):
    a = ["Z-score algorithm", "MinMax algorithm", "None"]


    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Normalizition"] == temp   ]

    npy[0,0] = Results_DF_all_mode["Mean_Accuracy_Left"].mean()  
    npy[0,1] = Results_DF_all_mode["Mean_Accuracy_Right"].mean()  
    npy[0,2] = Results_DF_all_mode["Mean_EER_Left"].mean()  
    npy[0,3] = Results_DF_all_mode["Mean_EER_Right"].mean()

    npy[1,0] = Results_DF_all_mode["Mean_Accuracy_Left"].min()  
    npy[1,1] = Results_DF_all_mode["Mean_Accuracy_Right"].min()  
    npy[1,2] = Results_DF_all_mode["Mean_EER_Left"].min()  
    npy[1,3] = Results_DF_all_mode["Mean_EER_Right"].min()  

    npy[2,0] = Results_DF_all_mode["Mean_Accuracy_Left"].max()  
    npy[2,1] = Results_DF_all_mode["Mean_Accuracy_Right"].max()  
    npy[2,2] = Results_DF_all_mode["Mean_EER_Left"].max()  
    npy[2,3] = Results_DF_all_mode["Mean_EER_Right"].max()    

    npy[3,0] = Results_DF_all_mode["Mean_Accuracy_Left"].median()  
    npy[3,1] = Results_DF_all_mode["Mean_Accuracy_Right"].median()  
    npy[3,2] = Results_DF_all_mode["Mean_EER_Left"].median()  
    npy[3,3] = Results_DF_all_mode["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])

    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())
    
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

PATH = os.path.join("Manuscripts", "src", "figures", "MinMax")
plt.tight_layout()
plt.savefig(PATH + "_ROC.png")
plt.close('all')


#########################################################################################################
#########################################################################################################
#########################################################################################################

plt.figure(figsize=(14,8))

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(test_ratios):
    a = ["20 percent", "35 percent", "50 percent"]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Test_Size"] == temp   ]

    npy[0,0] = Results_DF_all_mode["Mean_Accuracy_Left"].mean()  
    npy[0,1] = Results_DF_all_mode["Mean_Accuracy_Right"].mean()  
    npy[0,2] = Results_DF_all_mode["Mean_EER_Left"].mean()  
    npy[0,3] = Results_DF_all_mode["Mean_EER_Right"].mean()

    npy[1,0] = Results_DF_all_mode["Mean_Accuracy_Left"].min()  
    npy[1,1] = Results_DF_all_mode["Mean_Accuracy_Right"].min()  
    npy[1,2] = Results_DF_all_mode["Mean_EER_Left"].min()  
    npy[1,3] = Results_DF_all_mode["Mean_EER_Right"].min()  

    npy[2,0] = Results_DF_all_mode["Mean_Accuracy_Left"].max()  
    npy[2,1] = Results_DF_all_mode["Mean_Accuracy_Right"].max()  
    npy[2,2] = Results_DF_all_mode["Mean_EER_Left"].max()  
    npy[2,3] = Results_DF_all_mode["Mean_EER_Right"].max()    

    npy[3,0] = Results_DF_all_mode["Mean_Accuracy_Left"].median()  
    npy[3,1] = Results_DF_all_mode["Mean_Accuracy_Right"].median()  
    npy[3,2] = Results_DF_all_mode["Mean_EER_Left"].median()  
    npy[3,3] = Results_DF_all_mode["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])

    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())


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


PATH = os.path.join("Manuscripts", "src", "figures", "testsize")
plt.tight_layout()
plt.savefig(PATH + "_ROC.png")
plt.close('all')



#########################################################################################################
#########################################################################################################
#########################################################################################################
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
plt.figure(figsize=(14,8))

for idx, temp in enumerate(persentages):
    a = ["All Data", "Keeping 95 percent of variance"]
  
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["PCA"] == temp   ]

    npy[0,0] = Results_DF_all_mode["Mean_Accuracy_Left"].mean()  
    npy[0,1] = Results_DF_all_mode["Mean_Accuracy_Right"].mean()  
    npy[0,2] = Results_DF_all_mode["Mean_EER_Left"].mean()  
    npy[0,3] = Results_DF_all_mode["Mean_EER_Right"].mean()

    npy[1,0] = Results_DF_all_mode["Mean_Accuracy_Left"].min()  
    npy[1,1] = Results_DF_all_mode["Mean_Accuracy_Right"].min()  
    npy[1,2] = Results_DF_all_mode["Mean_EER_Left"].min()  
    npy[1,3] = Results_DF_all_mode["Mean_EER_Right"].min()  

    npy[2,0] = Results_DF_all_mode["Mean_Accuracy_Left"].max()  
    npy[2,1] = Results_DF_all_mode["Mean_Accuracy_Right"].max()  
    npy[2,2] = Results_DF_all_mode["Mean_EER_Left"].max()  
    npy[2,3] = Results_DF_all_mode["Mean_EER_Right"].max()    

    npy[3,0] = Results_DF_all_mode["Mean_Accuracy_Left"].median()  
    npy[3,1] = Results_DF_all_mode["Mean_Accuracy_Right"].median()  
    npy[3,2] = Results_DF_all_mode["Mean_EER_Left"].median()  
    npy[3,3] = Results_DF_all_mode["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])

    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())


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


PATH = os.path.join("Manuscripts", "src", "figures", "PCA")
plt.tight_layout()
plt.savefig(PATH + "_ROC.png")
plt.close('all')







#########################################################################################################
#########################################################################################################
#########################################################################################################

plt.figure(figsize=(14,8))
# plt.figure(figsize=(30,15))

FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
for idx, temp in enumerate(feature_names):
    a = ["All features", "Only MDIST features", "Only RDIST features", "Only TOTEX features", "Only MVELO features", "Only RANGE features", "Only AREAXX features", "Only MFREQ features", "Only FDPD features", "Only FDCX features"]    
    Results_DF_temp = Results_DF[   Results_DF["Features_Set"] == temp   ]
    Results_DF_temp = Results_DF_temp[   Results_DF_temp["PCA"] == 1   ]

    # print(Results_DF_temp.head())

    npy[0,0] =  Results_DF_temp["Mean_Accuracy_Left"].mean()  
    npy[0,1] =  Results_DF_temp["Mean_Accuracy_Right"].mean()  
    npy[0,2] =  Results_DF_temp["Mean_EER_Left"].mean()  
    npy[0,3] =  Results_DF_temp["Mean_EER_Right"].mean()

    npy[1,0] =  Results_DF_temp["Mean_Accuracy_Left"].min()  
    npy[1,1] =  Results_DF_temp["Mean_Accuracy_Right"].min()  
    npy[1,2] =  Results_DF_temp["Mean_EER_Left"].min()  
    npy[1,3] =  Results_DF_temp["Mean_EER_Right"].min()  

    npy[2,0] =  Results_DF_temp["Mean_Accuracy_Left"].max()  
    npy[2,1] =  Results_DF_temp["Mean_Accuracy_Right"].max()  
    npy[2,2] =  Results_DF_temp["Mean_EER_Left"].max()  
    npy[2,3] =  Results_DF_temp["Mean_EER_Right"].max()    

    npy[3,0] =  Results_DF_temp["Mean_Accuracy_Left"].median()  
    npy[3,1] =  Results_DF_temp["Mean_Accuracy_Right"].median()  
    npy[3,2] =  Results_DF_temp["Mean_EER_Left"].median()  
    npy[3,3] =  Results_DF_temp["Mean_EER_Right"].median()


    X = pd.DataFrame(npy, index=["mean", "min", "max", "median"] , columns=["Accuracy Left", "Accuracy Right", "EER Left", "EER Right"])




    with open(os.path.join("Manuscripts", "src", "tables", a[idx] + ".tex"), "w") as tf:
        tf.write(X.round(decimals=2).to_latex())

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

PATH = os.path.join("Manuscripts", "src", "figures", "feat")
plt.tight_layout()
plt.savefig(PATH + "_ROC.png")
plt.close('all')





