import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt
import sys, os, math
from scipy.spatial.distance import cdist
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



