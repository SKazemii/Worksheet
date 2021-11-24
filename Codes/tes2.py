import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys, os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join('..')))

from MLPackage.FS.ssa import jfs 


# load data
features_excel = "pfeatures"
working_path = os.getcwd()

feature_path = os.path.join(working_path, 'Datasets', features_excel + ".xlsx")
DF_features = pd.read_excel(feature_path, index_col = 0)





print( "[INFO] feature shape: ", DF_features.shape)


f_names = ['MDIST_RD', 'MDIST_AP', 'MDIST_ML', 'RDIST_RD', 'RDIST_AP', 'RDIST_ML', 'TOTEX_RD', 'TOTEX_AP', 'TOTEX_ML', 'MVELO_RD', 'MVELO_AP', 'MVELO_ML', 'RANGE_RD', 'RANGE_AP', 'RANGE_ML','AREA_CC', 'AREA_CE', 'AREA_SW', 'MFREQ_RD', 'MFREQ_AP', 'MFREQ_ML', 'FDPD_RD', 'FDPD_AP', 'FDPD_ML', 'FDCC', 'FDCE']
columnsName = f_names + [ "subject_ID", "left(0)/right(1)"]
DF_features.columns = columnsName

data  = pd.read_csv('C:\Project\Worksheet\Codes\ionosphere.csv')
print(DF_features.iloc[:,-2].unique())

data  = DF_features.values
feat  = np.asarray(data[:, 0:-2])
label = np.asarray(data[:, -2])

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of chromosomes
T    = 100   # maximum number of generations
CR   = 0.8
MR   = 0.01
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'CR':CR, 'MR':MR}

# perform feature selection
fmdl = jfs(feat, label, opts)

print(fmdl)
sys.exit()
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)

# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('GA')
ax.grid()
plt.show()
