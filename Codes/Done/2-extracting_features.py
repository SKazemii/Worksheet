
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os, logging, timeit
from pathlib import Path as Pathlb


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import itertools

PATH = os.path.join(os.getcwd(), "Codes")
if not PATH in sys.path:
    sys.path.append(PATH)


from MLPackage import Features as feat
from MLPackage.FS import hho
from MLPackage import config as cfg


project_dir = cfg.configs["paths"]["project_dir"]


fig_dir = cfg.configs["paths"]["fig_dir"]
tbl_dir = cfg.configs["paths"]["tbl_dir"]
results_dir = cfg.configs["paths"]["results_dir"]
temp_dir = cfg.configs["paths"]["temp_dir"]
log_path = cfg.configs["paths"]["log_path"]



def create_logger(level):
    loggerName = "main pynb"
    Pathlb(log_path).mkdir(parents=True, exist_ok=True)
    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    formatter_colored = logging.Formatter(blue + '[%(asctime)s]-' + yellow + '[%(name)s @%(lineno)d]' + reset + blue + '-[%(levelname)s]' + reset + bold_red + '\t\t%(message)s' + reset, datefmt='%m/%d/%Y %I:%M:%S %p ')
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @%(lineno)d]-[%(levelname)s]\t\t%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p ')
    file_handler = logging.FileHandler( os.path.join(log_path, loggerName + '_loger.log'), mode = 'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter_colored)


    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
logger = create_logger(logging.DEBUG)







eps = 5


data = np.load(cfg.configs["paths"]["casia_dataset.npy"])
metadata = np.load(cfg.configs["paths"]["casia_dataset-meta.npy"])
logger.info("Data shape: {}".format(data.shape))
logger.info("Metadata shape: {}".format(metadata.shape))



features = list()
prefeatures = list()


for sample, label in zip(data, metadata):
    
    COA = feat.computeCOATimeSeries(sample, Binarize = "simple", Threshold = 0)

    aMDIST = feat.computeMDIST(COA)    
    aRDIST = feat.computeRDIST(COA)
    aTOTEX = feat.computeTOTEX(COA)
    aMVELO = feat.computeMVELO(COA)
    aRANGE = feat.computeRANGE(COA)
    aAREACC = feat.computeAREACC(COA)
    aAREACE = feat.computeAREACE(COA)
    aAREASW = feat.computeAREASW(COA)
    aMFREQ = feat.computeMFREQ(COA)
    aFDPD = feat.computeFDPD(COA)
    aFDCC = feat.computeFDCC(COA)
    aFDCE = feat.computeFDCE(COA)

    handcraft_COAfeatures = np.concatenate((aMDIST, aRDIST, aTOTEX, aMVELO, aRANGE, [aAREACC], [aAREACE], [aAREASW], aMFREQ, aFDPD, [aFDCC], [aFDCE]), axis = 0)
    COAs = COA.flatten()

    GRF = feat.computeGRF(sample)
    handcraft_GRFfeatures = feat.computeGRFfeatures(GRF)

    wt_GRF = feat.wt_feature(GRF, waveletname="coif1", pywt_mode="constant", wavelet_level=4)

    wt_COA_RD = feat.wt_feature(COA[0,:], waveletname="coif1", pywt_mode="constant", wavelet_level=4)
    wt_COA_AP = feat.wt_feature(COA[1,:], waveletname="coif1", pywt_mode="constant", wavelet_level=4)
    wt_COA_ML = feat.wt_feature(COA[2,:], waveletname="coif1", pywt_mode="constant", wavelet_level=4)



    features.append( np.concatenate((COAs, handcraft_COAfeatures, GRF, handcraft_GRFfeatures, wt_COA_RD, wt_COA_AP, wt_COA_ML, wt_GRF, label[0:2]), axis=0)  )
    prefeatures.append(feat.prefeatures(sample))
    # break
    


columnsName = cfg.COA_RD + cfg.COA_AP + cfg.COA_ML + cfg.COA_HC + cfg.GRF + cfg.GRF_HC + cfg.wt_COA_RD + cfg.wt_COA_AP + cfg.wt_COA_ML + cfg.wt_GRF + cfg.label
pd.DataFrame(features, columns=columnsName).to_excel(cfg.configs["paths"]["casia_all_feature.xlsx"])
np.save(cfg.configs["paths"]["casia_image_feature.npy"], prefeatures)



DF_features_all = pd.read_excel(cfg.configs["paths"]["casia_all_feature.xlsx"], index_col = 0)


data  = DF_features_all.values
features  = np.asarray(data[:, 0:-2])
label = np.asarray(data[:, -2])









