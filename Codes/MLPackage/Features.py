import numpy as np
import pandas as pd
from scipy import ndimage, signal
import matplotlib.pyplot as plt
import sys, os, math
from scipy.spatial.distance import cdist

import pywt


from MLPackage import Butterworth
from MLPackage import convertGS2BW


def computeCOPTimeSeries(Footprint3D):
    """
    computeCOPTimeSeries(Footprint3D)
    Footprint3D : [x,y,t] image
    return COPTS : RD, AP, ML COP time series
    """

    ML = list()
    AP = list()

    for i in range(Footprint3D.shape[2]):
        temp = Footprint3D[:, :, i]
        temp2 = ndimage.measurements.center_of_mass(temp)
        ML.append(temp2[1])
        AP.append(temp2[0])
    
    lowpass = Butterworth.Butterworthfilter(mode= "lowpass", fs = 100, cutoff = 5, order = 4)
    ML = lowpass.filter(ML)
    AP = lowpass.filter(AP)

    ML_f = ML - np.mean(ML)
    AP_f = AP - np.mean(AP)

    a = ML_f ** 2
    b = AP_f ** 2
    RD_f = np.sqrt(a + b)

    COPTS = np.stack((RD_f, AP_f, ML_f), axis = 0)
    return COPTS


def computeCOATimeSeries(Footprint3D, Binarize = "otsu", Threshold = 1):

    """
    computeCOATimeSeries(Footprint3D)
    Footprint3D : [x,y,t] image
    Binarize = 'simple', 'otsu', 'adaptive'
    Threshold = 1
    return COATS : RD, AP, ML COA time series
    """
    GS2BW_object = convertGS2BW.convertGS2BW(mode = Binarize, TH = Threshold)
    aML = list()
    aAP = list()
    for i in range(Footprint3D.shape[2]):
        temp = Footprint3D[:, :, i]

        BW, threshold = GS2BW_object.GS2BW(temp)
        
        temp3 = ndimage.measurements.center_of_mass(BW)

        aML.append(temp3[1])
        aAP.append(temp3[0])


    lowpass = Butterworth.Butterworthfilter(mode= "lowpass", fs = 100, cutoff = 5, order = 4)
    aML = lowpass.filter(aML)
    aAP = lowpass.filter(aAP)
    aML_f = aML - np.mean(aML)
    aAP_f = aAP - np.mean(aAP)

    a = aML_f ** 2
    b = aAP_f ** 2
    aRD_f = np.sqrt(a + b)
    
    COATS = np.stack((aRD_f, aAP_f, aML_f), axis = 0)
    
    return COATS


def computeMDIST(COPTS):
    """
    computeMDIST(COPTS)
    MDIST : Mean Distance
    COPTS [3,t] : RD, AP, ML COP time series
    return MDIST [3] : [MDIST_RD, MDIST_AP, MDIST_ML]
    """
    
    MDIST = np.mean(np.abs(COPTS), axis=1)
    
    return MDIST


def computeRDIST(COPTS):
    """
    computeRDIST(COPTS)
    RDIST : RMS Distance
    COPTS [3,t] : RD, AP, ML COP time series
    return RDIST [3] : [RDIST_RD, RDIST_AP, RDIST_ML]
    """
    RDIST = np.sqrt(np.mean(COPTS ** 2,axis=1))
    
    return RDIST


def computeTOTEX(COPTS):
    """
    computeTOTEX(COPTS)
    TOTEX : Total Excursions
    COPTS [3,t] : RD, AP, ML COP time series
    return TOTEX [3] : TOTEX_RD, TOTEX_AP, TOTEX_ML    
    """
    
    TOTEX = list()
    TOTEX.append(np.sum(np.sqrt((np.diff(COPTS[2,:])**2)+(np.diff(COPTS[1,:])**2))))
    TOTEX.append(np.sum(np.abs(np.diff(COPTS[1,:]))))
    TOTEX.append(np.sum(np.abs(np.diff(COPTS[2,:]))))
    
    return TOTEX


def computeRANGE(COPTS):
    """
    computeRANGE(COPTS)
    RANGE : Range
    COPTS [3,t] : RD, AP, ML COP time series
    return RANGE [3] : RANGE_RD, RANGE_AP, RANGE_ML
    """
    RANGE = list()
    # print(cdist(COPTS[1:2,:].T, COPTS[1:2,:].T).shape)
    # print(cdist(COPTS[1:2,:].T, COPTS[1:2,:].T))
    # sys.exit()
    RANGE.append(np.max(cdist(COPTS[1:2,:].T, COPTS[1:2,:].T)))
    RANGE.append(np.max(COPTS[1,:])-np.min(COPTS[1,:]))
    RANGE.append(np.max(COPTS[2,:])-np.min(COPTS[2,:]))
    
    
    return RANGE


def computeMVELO(COPTS, T = 1):
    """
    computeMVELO(COPTS,varargin)
    MVELO : Mean Velocity
    COPTS [3,t] : RD, AP, ML COP time series
    T : the period of time selected for analysis (CASIA-D = 1s)
    return MVELO [3] : MVELO_RD, MVELO_AP, MVELO_ML
    """
    
    MVELO = list()
    MVELO.append((np.sum(np.sqrt((np.diff(COPTS[2,:])**2)+(np.diff(COPTS[1,:])**2))))/T)
    MVELO.append((np.sum(np.abs(np.diff(COPTS[1,:]))))/T)
    MVELO.append((np.sum(np.abs(np.diff(COPTS[2,:]))))/T)
    
    return MVELO


def computeAREACC(COPTS):
    """
    computeAREACC(COPTS)
    AREA-CC : 95% Confidence Circle Area
    COPTS [3,t] : RD (AP, ML) COP time series
    return AREACC [1] : AREA-CC
    """
    
    MDIST = computeMDIST(COPTS)
    RDIST = computeRDIST(COPTS)
    z05 = 1.645 # z.05 = the z statistic at the 95% confidence level
    SRD = np.sqrt((RDIST[0]**2)-(MDIST[0]**2)) #the standard deviation of the RD time series
    
    AREACC = np.pi*((MDIST[0]+(z05*SRD))**2)
    return AREACC


def computeAREACE(COPTS):
    """
    computeAREACE(COPTS)
    AREA-CE : 95% Confidence Ellipse Area
    COPTS [3,t] : (RD,) AP, ML COP time series
    return AREACE [1] : AREA-CE
    """
    
    F05 = 3
    RDIST = computeRDIST(COPTS)
    SAP = RDIST[1]
    SML = RDIST[2]
    SAPML = np.mean(COPTS[2,:]*COPTS[1,:])
    AREACE = 2*np.pi*F05*np.sqrt((SAP**2)*(SML**2)-(SAPML**2))

    return AREACE


def computeAREASW(COPTS,T = 1):
    """
    computeAREASW(COPTS, T)
    AREA-SW : Sway area
    COPTS [3,t] : RD, AP, ML COP time series
    T : the period of time selected for analysis (CASIA-D = 1s)
    return AREASW [1] : AREA-SW
    """
    
    AP = COPTS[1,:]
    ML = COPTS[2,:]

    AREASW = np.sum( np.abs((AP[1:]*ML[:-1])-(AP[:-1]*ML[1:])))/(2*T)
    
    return AREASW


def computeMFREQ(COPTS, T = 1):
    """
    computeMFREQ(COPTS, T)
    MFREQ : Mean Frequency
    COPTS [3,t] : RD, AP, ML COP time series
    T : the period of time selected for analysis (CASIA-D = 1s)
    return MFREQ [3] : MFREQ_RD, MFREQ_AP, MFREQ_ML
    """
    
    TOTEX = computeTOTEX(COPTS)
    MDIST = computeMDIST(COPTS)

    MFREQ = list()
    MFREQ.append( TOTEX[0]/(2*np.pi*T*MDIST[0]) )
    MFREQ.append( TOTEX[1]/(4*np.sqrt(2)*T*MDIST[1]))
    MFREQ.append( TOTEX[2]/(4*np.sqrt(2)*T*MDIST[2]))

    return MFREQ


def computeFDPD(COPTS):
    """
    computeFDPD(COPTS)
    FD-PD : Fractal Dimension based on the Plantar Diameter of the Curve
    COPTS [3,t] : RD, AP, ML COP time series
    return FDPD [3] : FD-PD_RD, FD-PD_AP, FD-PD_ML
    """

    N = COPTS.shape[1]
    TOTEX = computeTOTEX(COPTS)
    d = computeRANGE(COPTS)
    Nd = [elemt*N for elemt in d]
    dev = [i / j for i, j in zip(Nd, TOTEX)]
    
    
    FDPD = np.log(N)/np.log(dev)
    # sys.exit()
    return FDPD


def computeFDCC(COPTS):
    """
    computeFDCC(COPTS)
    FD-CC : Fractal Dimension based on the 95% Confidence Circle
    COPTS [3,t] : RD, (AP, ML) COP time series
    return FDCC [1] : FD-CC_RD
    """
    
    N = COPTS.shape[1]
    MDIST = computeMDIST(COPTS)    
    RDIST = computeRDIST(COPTS)
    z05 = 1.645; # z.05 = the z statistic at the 95% confidence level
    SRD = np.sqrt((RDIST[0]**2)-(MDIST[0]**2)) #the standard deviation of the RD time series

    d = 2*(MDIST[0]+z05*SRD)
    TOTEX = computeTOTEX(COPTS)
    
    FDCC = np.log(N)/np.log((N*d)/TOTEX[0])
    return FDCC


def computeFDCE(COPTS):
    """
    computeFDCE(COPTS)
    FD-CE : Fractal Dimension based on the 95% Confidence Ellipse
    COPTS [3,t] : (RD,) AP, ML COP time series
    return FDCE [2] : FD-CE_AP, FD-CE_ML
    """
    
    
    N = COPTS.shape[1]
    F05 = 3; 
    RDIST = computeRDIST(COPTS)
    SAPML = np.mean(COPTS[2,:]*COPTS[1,:])
    
    d = np.sqrt(8*F05*np.sqrt(((RDIST[1]**2)*(RDIST[2]**2))-(SAPML**2)))
    TOTEX = computeTOTEX(COPTS)

    FDCE = np.log(N)/np.log((N*d)/TOTEX[0])
    
    return FDCE


def computeGRF(Footprint3D):
    """
    computeGFR(Footprint3D)
    Footprint3D: [x,y,t] image
    return GFR: [t] time series signal
    """
    GRF = list()
    

    for frame in range(Footprint3D.shape[2]):
        temp = Footprint3D[:, :, frame].sum()
        GRF.append(temp)
    
    return GRF


def computeGRFfeatures(GRF):
    """
    computeGRFfeatures(GRF)
    GRF: [t] time series signal
    return GFR features: [9] (max_value_1, max_value_1_ind, max_value_2, max_value_2_ind, min_value, min_value_ind, mean_value, std_value, sum_value)
    """
    # handcraft_features = list()
    L = int(len(GRF)/2)




    max_value_1 = np.max(GRF[:L])
    max_value_1_ind = np.argmax( GRF[:L] )
    max_value_2 = np.max(GRF[L:])
    max_value_2_ind = L + np.argmax(	GRF[L:] )

    min_value = np.min(GRF[max_value_1_ind:max_value_2_ind])
    min_value_ind = max_value_1_ind + np.argmin(	GRF[max_value_1_ind:max_value_2_ind] )

    mean_value = np.mean(GRF)
    std_value = np.std(GRF)
    sum_value = np.sum(GRF)

    handcraft_features = [max_value_1, max_value_1_ind,
                    max_value_2, max_value_2_ind,
                    min_value, min_value_ind,
                    mean_value, std_value, sum_value]

    return handcraft_features


def wt_feature(signal, waveletname="coif1", pywt_mode="constant", wavelet_level=4):
    """
    wt_feature(signal, waveletname, pywt_mode, wavelet_level)
    signal: [t] time series signal
    wavelet_level = 4 or pywt.dwt_max_level(100, waveletname)
	pywt_mode = "constant"
    waveletname = "coif1"
        haar family: haar
        db family: db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17, db18, db19, db20, db21, db22, db23, db24, db25, db26, db27, db28, db29, db30, db31, db32, db33, db34, db35, db36, db37, db38
        sym family: sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20   
        coif family: coif1, coif2, coif3, coif4, coif5, coif6, coif7, coif8, coif9, coif10, coif11, coif12, coif13, coif14, coif15, coif16, coif17bior family: bior1.1, bior1.3, bior1.5, bior2.2, bior2.4, bior2.6, bior2.8, bior3.1, bior3.3, bior3.5, bior3.7, bior3.9, bior4.4, bior5.5, bior6.8
        rbio family: rbio1.1, rbio1.3, rbio1.5, rbio2.2, rbio2.4, rbio2.6, rbio2.8, rbio3.1, rbio3.3, rbio3.5, rbio3.7, rbio3.9, rbio4.4, rbio5.5, rbio6.8
        dmey family: dmey
        gaus family: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
        mexh family: mexh
        morl family: morl
        cgau family: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
        shan family: shan
        fbsp family: fbsp
        cmor family: cmor
	

    return dwt_coeff: Discrete Wavelet Transform coeff
    """

    dwt_coeff = pywt.wavedec(signal, waveletname, mode=pywt_mode, level=wavelet_level)
    dwt_coeff = np.concatenate(dwt_coeff).ravel()

    return dwt_coeff


def prefeatures(Footprint3D, eps=5):
    """
    prefeatures(Footprint3D)
    Footprint3D: [x,y,t] image
    return prefeatures: [x, y, 10] (CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100)

    If The 30th percentile of a is 24.0: This means that 30% of values fall below 24.
    
    """

    prefeatures = list()
        
    temp = np.zeros(Footprint3D.shape)
    temp[Footprint3D > eps] = 1
    CD = np.sum(temp, axis=2)

    PTI = np.sum(Footprint3D, axis=2)

    Tmax = np.argmax(Footprint3D, axis=2)
        
    temp = Footprint3D.copy()
    temp[Footprint3D < eps] = 0
    x = np.ma.masked_array(temp, mask=temp==0)
    Tmin = np.argmin(x , axis=2, )

    P50  = np.percentile(Footprint3D,  50, axis=2)
    P60  = np.percentile(Footprint3D,  60, axis=2)
    P70  = np.percentile(Footprint3D,  70, axis=2)
    P80  = np.percentile(Footprint3D,  80, axis=2)
    P90  = np.percentile(Footprint3D,  90, axis=2)
    P100 = np.percentile(Footprint3D, 100, axis=2)

    prefeatures = np.stack((CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100), axis = -1)

    return prefeatures