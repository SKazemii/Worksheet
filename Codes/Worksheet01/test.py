import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import ROC_plot as perf
from MLPackage import Features as fe
####################################################
# COPTS = np.load("./Codes/Worksheet01/ToonCodes/COP.npy")
# f3D = np.load("./Codes/Worksheet01/ToonCodes/3D.npy")
# np.seterr('raise')

# print(fe.computeFDCE(COPTS))







####################################################
EER = np.load("./Datasets/EER.npy")
FPR = np.load("./Datasets/FPR.npy")
FNR = np.load("./Datasets/FNR.npy")

far = (np.mean(FPR, axis=0))
print(np.mean(EER))
frr = (np.mean(FNR, axis=0))
THRESHOLDs = np.linspace(0, 300, 10)

perf.ROC_plot_v2(far, frr, THRESHOLDs, "./")
plt.show()



# %computeCFREQ
# %compute95FREQ
# %computeFREQD
# %computeMEDFREQ
# %computePOWER
####################################################