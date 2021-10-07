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
data_dir = os.path.join(project_dir, "results")

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
test_ratios = [0.2]
persentages = [0.95]
Modes = ["corr"]
model_types = ["min"]
THRESHOLDs = np.linspace(0, 1, 100)
normilizings = ["z-score"]
feature_names = ["MDIST"]

color = ['darkorange', 'navy', 'red', 'greenyellow', 'lightsteelblue', 'lightcoral', 'olive', 'mediumpurple', 'khaki', 'hotpink']





# Results_DF_all = Results_DF[   Results_DF["Features_Set"] == "MDIST"   ]
auc=[]
plt.figure(figsize=(14,8))
npy = np.empty((4,4))
FAR_L = list()
FRR_L = list()
FAR_R = list()
FRR_R = list()
Results_DF_all_mode = Results_DF[   Results_DF["Mode"] == "corr"   ]
Results_DF_all_mode = Results_DF_all_mode[   Results_DF_all_mode["Features_Set"] == "MDIST"   ]
Results_DF_all_mode = Results_DF_all_mode[   Results_DF_all_mode["Normalizition"] == "z-score"   ]
Results_DF_all_mode = Results_DF_all_mode[   Results_DF_all_mode["Model_Type"] == "min"   ]
Results_DF_all_mode = Results_DF_all_mode[   Results_DF_all_mode["Test_Size"] == 0.2   ]
Results_DF_all_mode = Results_DF_all_mode[   Results_DF_all_mode["PCA"] == 0.95   ]

print(Results_DF_all_mode.head())

cols = ["FAR_L_" + str(i) for i in range(100)] 
FAR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

cols = ["FRR_L_" + str(i) for i in range(100)]
FRR_L.append(Results_DF_all_mode.loc[:, cols].mean().values)

cols = ["FAR_R_" + str(i) for i in range(100)] 
FAR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

cols = ["FRR_R_" + str(i) for i in range(100)]
FRR_R.append(Results_DF_all_mode.loc[:, cols].mean().values)

print(FRR_L)
print(FRR_R)
plt.subplot(1,2,1)
auc = (1 + np.trapz( FRR_L, FAR_L))
label="jj" #+ ' AUC = ' + str(round(auc, 2))

plt.plot(FAR_L, FRR_L, linestyle='--', marker='o', color=color[0], lw = 2, label="jj", clip_on=False)

plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Acceptance Rate')
plt.ylabel('False Rejection Rate')
plt.title('ROC curve, left side')
plt.gca().set_aspect('equal')
plt.legend(loc="best")

plt.subplot(1,2,2)

auc = (1 + np.trapz( FRR_R, FAR_R))
plt.plot(FAR_R, FRR_R, linestyle='--', marker='o', color=color[0], lw = 2, label="a[idx]", clip_on=False)



plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Acceptance Rate')
plt.ylabel('False Rejection Rate')
plt.title('ROC curve, Right side')
plt.gca().set_aspect('equal')
plt.legend(loc="best")
plt.show()
sys.exit()

for idx, temp in enumerate(Modes):
    a = ["Correlation", "Euclidean Distance"]
    

    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Mode"] == temp   ]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Features_Set"] == "MDIST"   ]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Normalizition"] == "z-score"   ]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Model_Type"] == "min"   ]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["Test_Size"] == 0.2   ]
    Results_DF_all_mode = Results_DF_all[   Results_DF_all["PCA"] == 0.95   ]

    print(Results_DF_all_mode.head())
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

    
    






   

PATH = os.path.join("Manuscripts", "src", "figures", "Correlation")
plt.tight_layout()
plt.savefig(PATH + "_ROC.png")
plt.close('all')




