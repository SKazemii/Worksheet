import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

from itertools import combinations, product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import util as perf


test_ratio = [0.2, 0.35, 0.5]
persentage = [1.0, 0.95]
mode = ["corr", "dist"]
model_type = ["min", "median", "average"]
THRESHOLDs = np.linspace(0, 1, 100)
score = None#"B"
score = "A"#"B"
normilizing = ["z-score", "minmax", "None"]

feature_names = ["All", "MDIST", "RDIST", "TOTEX", "MVELO", "RANGE", "AREAXX", "MFREQ", "FDPD", "FDCX"]



output = list(product(mode, model_type, persentage, test_ratio, normilizing, feature_names ))

print(output)
print(len(output))
for i in output:
    print(i[0]+ "_" + i[1] + "_" + str(i[2])+ "_" + str(i[3]) + "_" + i[4] + "_" + i[5])
    folder = i[0]+ "_" + i[1] + "_" + str(i[2])+ "_" + str(i[3]) + "_" + i[4] + "_" + i[5]
    
    path =  "/Users/saeedkazemi/Documents/Python/Worksheet/results/" + folder + "/NPY/"
    print(path)

    EER_R = np.load(path + "EER_R.npy")
    FAR_R = np.load(path + "FAR_R.npy")
    FRR_R = np.load(path + "FRR_R.npy")
    EER_L = np.load(path + "EER_L.npy")
    FRR_L = np.load(path + "FRR_L.npy")
    FAR_L = np.load(path + "FAR_L.npy")
    ACC_L = np.load(path + "ACC_L.npy")
    ACC_R = np.load(path + "ACC_R.npy")
    
    print(ACC_R)
    print(ACC_R.shape)
    sys.exit()

ss= 2*3*2*3*3*10
print(ss)