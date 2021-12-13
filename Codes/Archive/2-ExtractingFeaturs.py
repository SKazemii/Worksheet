import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import sys, os, multiprocessing
from scipy.spatial import distance


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLPackage import Features as fe

working_path = os.getcwd()

print("[INFO] OS: ", sys.platform)
print("[INFO] Core Number: ", multiprocessing.cpu_count())

data_path = os.path.join(working_path, 'Datasets', 'datalist.npy')
meta_path = os.path.join(working_path, 'Datasets', 'metadatalist.npy')


data = np.load(data_path)
metadata = np.load(meta_path)
print("[INFO] data shape: ", data.shape)
print("[INFO] metadata shape: ", metadata.shape)

pfeatures = list()
COPs = list()

afeatures_simple = list()
COAs_simple = list()
afeatures_otsu = list()
COAs_otsu = list()

for j in range(data.shape[0]):
    COPTS = fe.computeCOPTimeSeries(data[j])
    COATS_simple = fe.computeCOATimeSeries(data[j], Binarize = "simple", Threshold = 0)
    COATS_otsu = fe.computeCOATimeSeries(data[j], Binarize = "otsu")

    pMDIST = fe.computeMDIST(COPTS) 
    pRDIST = fe.computeRDIST(COPTS)
    pTOTEX = fe.computeTOTEX(COPTS)
    pMVELO = fe.computeMVELO(COPTS)
    pRANGE = fe.computeRANGE(COPTS)
    pAREACC = fe.computeAREACC(COPTS)
    pAREACE = fe.computeAREACE(COPTS)
    pAREASW = fe.computeAREASW(COPTS)
    pMFREQ = fe.computeMFREQ(COPTS)
    pFDPD = fe.computeFDPD(COPTS)
    pFDCC = fe.computeFDCC(COPTS)
    pFDCE = fe.computeFDCE(COPTS)
    pfeatures.append(np.concatenate((pMDIST, pRDIST, pTOTEX, pMVELO, pRANGE, [pAREACC], [pAREACE], [pAREASW], pMFREQ, pFDPD, [pFDCC], [pFDCE], metadata[j,0:2]), axis = 0) )
    COPs.append(np.concatenate((COPTS.flatten(), metadata[j,0:2]), axis = 0))


    aMDIST_simple = fe.computeMDIST(COATS_simple)    
    aRDIST_simple = fe.computeRDIST(COATS_simple)
    aTOTEX_simple = fe.computeTOTEX(COATS_simple)
    aMVELO_simple = fe.computeMVELO(COATS_simple)
    aRANGE_simple = fe.computeRANGE(COATS_simple)
    aAREACC_simple = fe.computeAREACC(COATS_simple)
    aAREACE_simple = fe.computeAREACE(COATS_simple)
    aAREASW_simple = fe.computeAREASW(COATS_simple)
    aMFREQ_simple = fe.computeMFREQ(COATS_simple)
    aFDPD_simple = fe.computeFDPD(COATS_simple)
    aFDCC_simple = fe.computeFDCC(COATS_simple)
    aFDCE_simple = fe.computeFDCE(COATS_simple)
    afeatures_simple.append(np.concatenate((aMDIST_simple, aRDIST_simple, aTOTEX_simple, aMVELO_simple, aRANGE_simple, [aAREACC_simple], [aAREACE_simple], [aAREASW_simple], aMFREQ_simple, aFDPD_simple, [aFDCC_simple], [aFDCE_simple], metadata[j,0:2]), axis = 0) )
    COAs_simple.append(np.concatenate((COATS_simple.flatten(), metadata[j,0:2]), axis = 0))

    aMDIST_otsu = fe.computeMDIST(COATS_otsu)    
    aRDIST_otsu = fe.computeRDIST(COATS_otsu)
    aTOTEX_otsu = fe.computeTOTEX(COATS_otsu)
    aMVELO_otsu = fe.computeMVELO(COATS_otsu)
    aRANGE_otsu = fe.computeRANGE(COATS_otsu)
    aAREACC_otsu = fe.computeAREACC(COATS_otsu)
    aAREACE_otsu = fe.computeAREACE(COATS_otsu)
    aAREASW_otsu = fe.computeAREASW(COATS_otsu)
    aMFREQ_otsu = fe.computeMFREQ(COATS_otsu)
    aFDPD_otsu = fe.computeFDPD(COATS_otsu)
    aFDCC_otsu = fe.computeFDCC(COATS_otsu)
    aFDCE_otsu = fe.computeFDCE(COATS_otsu)
    afeatures_otsu.append(np.concatenate((aMDIST_otsu, aRDIST_otsu, aTOTEX_otsu, aMVELO_otsu, aRANGE_otsu, [aAREACC_otsu], [aAREACE_otsu], [aAREASW_otsu],   aMFREQ_otsu, aFDPD_otsu, [aFDCC_otsu], [aFDCE_otsu], metadata[j,0:2]), axis = 0) )
    COAs_otsu.append(np.concatenate((COATS_otsu.flatten(), metadata[j,0:2]), axis = 0))

    #
        # print(pFDPD)
        # print(pFDCE)
        # print(pFDCC)

        # sys.exit()

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
    
    
    

    
    

saving_path = os.path.join(working_path, 'Datasets', 'pfeatures.xlsx')
columnsName = ["feature_" + str(i) for i in range(len(pfeatures[0])-2)] + [ "subject ID", "left(0)/right(1)"]
pd.DataFrame(pfeatures, columns=columnsName).to_excel(saving_path)


saving_path = os.path.join(working_path, 'Datasets', 'afeatures_simple.xlsx')
columnsName = ["feature_" + str(i) for i in range(len(afeatures_simple[0])-2)] + [ "subject ID", "left(0)/right(1)"]
pd.DataFrame(afeatures_simple, columns=columnsName).to_excel(saving_path)


saving_path = os.path.join(working_path, 'Datasets', 'afeatures_otsu.xlsx')
columnsName = ["feature_" + str(i) for i in range(len(afeatures_otsu[0])-2)] + [ "subject ID", "left(0)/right(1)"]
pd.DataFrame(afeatures_otsu, columns=columnsName).to_excel(saving_path)



saving_path = os.path.join(working_path, 'Datasets', 'COPs.xlsx')
columnsName = ["feature_" + str(i) for i in range(len(COPs[0])-2)] + [ "subject ID", "left(0)/right(1)"]
pd.DataFrame(COPs, columns=columnsName).to_excel(saving_path)


saving_path = os.path.join(working_path, 'Datasets', 'COAs_simple.xlsx')
columnsName = ["feature_" + str(i) for i in range(len(COAs_simple[0])-2)] + [ "subject ID", "left(0)/right(1)"]
pd.DataFrame(COAs_simple, columns=columnsName).to_excel(saving_path)


saving_path = os.path.join(working_path, 'Datasets', 'COAs_otsu.xlsx')
columnsName = ["feature_" + str(i) for i in range(len(COAs_otsu[0])-2)] + [ "subject ID", "left(0)/right(1)"]
pd.DataFrame(COAs_otsu, columns=columnsName).to_excel(saving_path)

print("[INFO] Done!!!")




