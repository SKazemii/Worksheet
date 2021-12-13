import warnings

from numpy.core.einsumfunc import _update_other_results
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os, logging, timeit, pprint, copy
from pathlib import Path as Pathlb


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import multiprocessing

from itertools import product


import sklearn as sk
import tensorflow as tf
from tensorflow import keras

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning


from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection








sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    
from MLPackage import config as cfg
from MLPackage import Utilities as util


project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "Manuscripts", "src", "figures")
tbl_dir = os.path.join(project_dir, "Manuscripts", "src", "tables")
results_dir = os.path.join(project_dir, "results")
dataset_dir = os.path.join(project_dir, "Datasets")
temp_dir = os.path.join(project_dir, "temp")
log_path = os.path.join(project_dir, 'logs')

Pathlb(log_path).mkdir(parents=True, exist_ok=True)
Pathlb(dataset_dir).mkdir(parents=True, exist_ok=True)
Pathlb(temp_dir).mkdir(parents=True, exist_ok=True)
Pathlb(results_dir ).mkdir(parents=True, exist_ok=True)
Pathlb(fig_dir).mkdir(parents=True, exist_ok=True)
Pathlb(tbl_dir).mkdir(parents=True, exist_ok=True)




def create_logger(level):
    loggerName = Pathlb(__file__).stem
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




time = int(timeit.default_timer() * 1_000_000)

def collect_results(result):
    global time
    excel_path = cfg.configs["paths"]["results_dir"]

    if os.path.isfile(excel_path):
        Results_DF = pd.read_excel(excel_path, index_col = 0)
    else:
        Results_DF = pd.DataFrame(columns=util.columnsname)

    Results_DF = Results_DF.append(result)
    try:
        Results_DF.to_excel(excel_path, columns=util.columnsname)
    except:
        Results_DF.to_excel(os.path.join(os.getcwd(), 'temp', 'Results'+str(time)+'.xlsx'), columns=util.columnsname)




def main():

    CNNs = ["vgg16.VGG16", "resnet50.ResNet50", "efficientnet.EfficientNetB0", "mobilenet.MobileNet"]
    feat_Name = ["PTI", "Tmax", "Tmin", "P50", "P60", "P70", "P80", "P90", "P100"]

    for network in CNNs:
        file_name =  network.split(".")[0] + '_CD_features.xlsx'
        loading_path = os.path.join(cfg.configs['paths']["casia_deep_feature"], file_name)
        
        a = pd.read_excel(loading_path,  index_col = 0)
        c = a.iloc[:, -2:]
        c.columns = cfg.label

        a = a.iloc[:, :-2]
        columns = ["CD_"+str(i) for i in range(a.shape[1])]
        a.columns = columns

        for f in feat_Name:
            file_name =  network.split(".")[0] + '_' + f +'_features.xlsx'
            loading_path = os.path.join(cfg.configs['paths']["casia_deep_feature"], file_name)

            b = pd.read_excel(loading_path,  index_col = 0)
            b = b.iloc[:, :-2]
            columns = [f+"_"+str(i) for i in range(b.shape[1])]
            b.columns = columns
        
            a = pd.concat([a, b], axis=1)


        a = pd.concat([a, c], axis=1)


        file_name =  network.split(".")[0] + '_fusion_features.csv'
        
        try:
            pd.DataFrame(a).to_csv(os.path.join(cfg.configs['paths']["casia_deep_feature"], file_name))
        except Exception as e:
            print(e)
            pd.DataFrame(a).to_csv(os.path.join(cfg.configs['paths']["casia_deep_feature"], file_name+str(time)+'.csv'))
        finally:
            logger.info(f"saving fusion features of {network.split('.')[0]} with shape of {a.shape}!!!")


    logger.info("Done!!!")



if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))
