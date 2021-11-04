import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from keras.applications.mobilenet import MobileNet, preprocess_input


plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["figure.dpi"] = 128


classifier_name = "tree"  # {lda, knn, svm, reg, tree}
features_name = "Mobnet"  # {temporal, statistical, spectral, all, AR, vgg16, Mobnet}
transform = "standardization"  # {standardization, normalization, none}
# already done = mobilenet | inceptionresnetv2 | xception | inceptionv3 | resnet50 | vgg16 | vgg19 | spatial
model_name = "mobilenet"


# weights = {none | imagenet}
weights = "imagenet"
# include_top = {True | False}
include_top = False


VarianceThresholdflag = True
Highcorrelatedflag = False

test_size = 0.2
seed = 10
Grid_n_jobs = 3
Grid_refit = True

# outer_n_splits = 10
# outer_shuffle = True

inner_n_splits = 10
inner_shuffle = True


# define search spaces
knnspace1 = {
    "model__n_neighbors": np.arange(1, 22, 2),
    "model__metric": ["euclidean", "manhattan", "chebyshev"],
    "model__weights": ["distance", "uniform"],
    "sfs__k_features": [100, 200, 300],
}
knnspace = {
    "n_neighbors": [5, 11, 15, 19, 23],
    "metric": ["euclidean", "manhattan"],
    "weights": ["distance", "uniform"],
    # "sfs__k_features": [100, 200, 300],
}

# knnspace = [
#     {"n_neighbors": np.arange(1, 30, 2), "metric": ["euclidean", "manhattan", "chebyshev"], "weights": ["distance", "uniform"],},
#     {"n_neighbors": np.arange(1, 30, 2), "metric": ["mahalanobis", "seuclidean"],
#      "metric_params": [{"V": np.cov(X_train)}],"weights": ["distance", "uniform"],}
# ]

treespace1 = {
    "model__max_depth": np.arange(3, 33, 2),
    "model__criterion": ["gini", "entropy"],
    "sfs__k_features": [100, 200, 300],
}
treespace = {
    "max_depth": np.arange(6, 18, 2),
    "criterion": ["gini", "entropy"],
}


svmspace1 = {
    "model__probability": [True],
    "model__kernel": ["rbf", "linear"],
    "model__decision_function_shape": ["ovr", "ovo"],
    "model__C": [0.1, 10, 1000],
    "model__gamma": [1, 0.01, 0.0001],
    "model__random_state": [seed],
    "sfs__k_features": [100, 200, 300],
}
svmspace = {
    "probability": [True],
    "kernel": ["linear", "rbf"],
    # "C": [0.1, 10, 1000],
    # "gamma": [1, 0.01, 0.0001],
    "random_state": [seed],
}

ldaspace = {
    # "n_components": [10, 20, 30],
    # "model__n_components": [10, 20, 30]
    # "sfs__k_features": [100],
}

regspace = {
    "C": [1, 10, 100, 1000],
}

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "temp", "src", "figures", "project")
tbl_dir = os.path.join(project_dir, "temp", "src", "tables", "project")
data_dir = os.path.join(project_dir, "Dataset", "project", "Step Scan Dataset", "[H5]")
dataset_file = os.path.join(data_dir, "footpressures_align.h5")
pickle_dir = os.path.join(project_dir, "Dataset", "project", "pickle")
output_dir = os.path.join(project_dir, "temp", "output")


result_name_file = (
    datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    + "-"
    + classifier_name
    + "-"
    + features_name
    + "-"
    + transform
)

# ============ RC model configuration and hyperparameter values ============
