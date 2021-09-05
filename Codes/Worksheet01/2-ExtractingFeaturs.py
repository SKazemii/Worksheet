import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os
from scipy.spatial import distance


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import Features as fe



data = np.load("./Datasets/datalist.npy")
metadata = np.load("./Datasets/metadatalist.npy")
print("[INFO] data shape: ", data.shape)
print("[INFO] metadata shape: ",metadata.shape)

features = list()

for j in range(data.shape[0]):
    COPTS = fe.computeCOPTimeSeries(data[j])
    COATS = fe.computeCOATimeSeries(data[j], Binarize = "simple", Threshold = 1)

    pMDIST = fe.computeMDIST(COPTS)    
    aMDIST = fe.computeMDIST(COATS)    
    
    pRDIST = fe.computeRDIST(COPTS)
    aRDIST = fe.computeRDIST(COATS)

    pTOTEX = fe.computeTOTEX(COPTS)
    aTOTEX = fe.computeTOTEX(COATS)

    pMVELO = fe.computeMVELO(COPTS)
    aMVELO = fe.computeMVELO(COATS)

    pRANGE = fe.computeRANGE(COPTS)
    aRANGE = fe.computeRANGE(COATS)

    pAREACC = fe.computeAREACC(COPTS)
    aAREACC = fe.computeAREACC(COATS)

    pAREACE = fe.computeAREACE(COPTS)
    aAREACE = fe.computeAREACE(COATS)

    pAREASW = fe.computeAREASW(COPTS)
    aAREASW = fe.computeAREASW(COATS)

    pMFREQ = fe.computeMFREQ(COPTS)
    aMFREQ = fe.computeMFREQ(COATS)

    pFDPD = fe.computeFDPD(COPTS)
    aFDPD = fe.computeFDPD(COATS)

    pFDCC = fe.computeFDCC(COPTS)
    aFDCC = fe.computeFDCC(COATS)

    pFDCE = fe.computeFDCE(COPTS)
    aFDCE = fe.computeFDCE(COATS)

    # sys.exit()

    
    # print(pFDCE)
    # plt.figure()
    # plt.plot(range(100), COPTS[2])
    # plt.figure()
    # plt.plot(range(100), COPTS[1])
    # plt.figure()
    # plt.plot(range(100), COPTS[0])
    
    # plt.figure()
    # plt.plot(range(100), COATS[2])
    # plt.figure()
    # plt.plot(range(100), COATS[1])
    # plt.figure()
    # plt.plot(range(100), COATS[0])


    # plt.show()
    
    features.append(np.concatenate((pMDIST, aMDIST, pRDIST, aRDIST, pTOTEX, aTOTEX, pMVELO, aMVELO, pRANGE, aRANGE, [pAREACC], [aAREACC], [pAREACE], [aAREACE], pMFREQ, aMFREQ, pFDPD, aFDPD, [pFDCC], [aFDCC], [pFDCE], [aFDCE], metadata[j,0:2]), axis = 0) )
    


np.save("./Datasets/features.npy", features)
print("[INFO] Done!!!")





