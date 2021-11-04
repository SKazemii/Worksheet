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
    logger.info("OS: {}".format( sys.platform))
    logger.info("Core Number: {}".format(multiprocessing.cpu_count()))


    data = pd.read_excel(os.path.join(project_dir, 'temp', 'GRF.xlsx'), index_col = 0)

    scaling = preprocessing.StandardScaler()
    Scaled_train = scaling.fit_transform(data.iloc[:, :-2])

    logger.info("data shape: {}".format(Scaled_train.shape))
    # logger.info("data: \n{}\n".format(data))

    # pca = PCA(random_state = 2019)
    # X_pca = pca.fit_transform(Scaled_train.iloc[:,:-2])
    # print(X_pca)

    
    # n_batches = 16
    # inc_pca = IncrementalPCA()
    # for X_batch in np.array_split(Scaled_train, n_batches):
    #     inc_pca.partial_fit(X_batch)
    # X_ipca = inc_pca.transform(Scaled_train)
    # print(X_ipca)
    # print(X_ipca.shape)

    
    # kpca = KernelPCA(kernel="rbf", n_components=100, gamma=None, fit_inverse_transform=True, random_state = 2019)
    # kpca.fit(Scaled_train)
    # X_kpca = kpca.transform(Scaled_train)
    # print(X_kpca)
    # print(X_kpca.shape)


    # sparsepca = SparsePCA(alpha=0.0001, random_state=2019, n_jobs=-1)
    # sparsepca.fit(Scaled_train)
    # X_spacepca = sparsepca.transform(Scaled_train)
    # print(X_spacepca)
    # print(X_spacepca.shape)
    
    lle = LocallyLinearEmbedding(n_neighbors = 10, method = 'modified', n_jobs = -1,  random_state=2019)
    lle.fit(Scaled_train)
    X_lle = lle.transform(Scaled_train)
    print(X_lle)
    print(X_lle.shape)
    plt

    tsne = TSNE(n_components=2,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
    X_tsne = tsne.fit_transform(Scaled_train)


    plt.figure(figsize=(12,12))
    data.iloc[:, -2]

    plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='red', alpha=0.5,label='0')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='blue', alpha=0.5,label='1')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='green', alpha=0.5,label='2')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='black', alpha=0.5,label='3')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='khaki', alpha=0.5,label='4')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='yellow', alpha=0.5,label='5')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='turquoise', alpha=0.5,label='6')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='pink', alpha=0.5,label='7')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='moccasin', alpha=0.5,label='8')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='olive', alpha=0.5,label='9')
    # plt.scatter(X_lle[data.iloc[:, -2], 0], X_lle[data.iloc[:, -2], 1], color='coral', alpha=0.5,label='10')
    plt.title("PCA")
    plt.ylabel('Les coordonnees de Y')
    plt.xlabel('Les coordonnees de X')
    plt.legend()
    plt.show()

    logger.info("Done!!!")



if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))

