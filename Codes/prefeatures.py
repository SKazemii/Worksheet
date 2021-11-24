import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import sys, os, logging, timeit
from pathlib import Path as Pathlb


from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import itertools, multiprocessing




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
from MLPackage import util as ut
from MLPackage import Butterworth


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

    logger = logging.getLogger(loggerName)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s]-[%(name)s @ %(lineno)d]-[%(levelname)s]\t%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    file_handler = logging.FileHandler( os.path.join(log_path, loggerName + '_loger.log'), mode = 'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
logger = create_logger(logging.DEBUG)




def projection(PCA):
    pass



def main():
    data_path = os.path.join(project_dir, 'Datasets', 'datalist.npy')
    meta_path = os.path.join(project_dir, 'Datasets', 'metadatalist.npy')
    eps = 5


    data = np.load(data_path)
    metadata = np.load(meta_path)
    logger.info("Data shape: {}".format(data.shape))
    logger.info("Metadata shape: {}".format(metadata.shape))

    prefeatures = list()
    for j in range(data.shape[0]):
        
        I = np.zeros((data[j].shape))
        I[data[j] > eps] = 1
        CD = np.sum(I, axis=2)

        PTI = np.sum(data[j], axis=2)

        Tmax = np.argmax(data[j], axis=2)
        
        I = data[j].copy()
        I[data[j] < eps] = 0
        x = np.ma.masked_array(I, mask=I==0)
        Tmin = np.argmin(x , axis=2, )



        P50  = np.percentile(data[j],  50, axis=2)
        P60  = np.percentile(data[j],  60, axis=2)
        P70  = np.percentile(data[j],  70, axis=2)
        P80  = np.percentile(data[j],  80, axis=2)
        P90  = np.percentile(data[j],  90, axis=2)
        P100 = np.percentile(data[j], 100, axis=2)

        prefeatures.append(np.stack((CD, PTI, Tmin, Tmax, P50, P60, P70, P80, P90, P100), axis = -1))

        logger.info(prefeatures[0].shape)
        plt.figure()
        plt.imshow(prefeatures[:, :, 4])
        plt.show()
        sys.exit()

    saving_path = os.path.join(project_dir, 'Datasets', 'prefeatures.npy')
    np.save(saving_path, prefeatures)

    temp = np.load(saving_path)
    logger.info("prefeature shape:".format(temp.shape))




if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))

