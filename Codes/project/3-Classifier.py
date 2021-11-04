from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.feature_selection import VarianceThreshold
import sklearn
from collections import Counter


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import sklearn.model_selection
from sklearn import preprocessing
from sklearn import feature_selection

import seaborn as sns

# import tsfel
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# Custom imports
import config as cfg


def interpolate(inp, fi):
    i, f = (
        int(fi // 1),
        fi % 1,
    )  # Split floating-point index into whole & fractional parts.
    j = i + 1 if round(f, 4) > 0 else i  # Avoid index error.
    return (1 - f) * inp[i] + f * inp[j]


print("[INFO] importing libraries....")


# df_sum ##################################################################################
print("[INFO] importing pickles files....")
with open(
    os.path.join(cfg.pickle_dir, "df_sum_temporal_features.pickle"), "rb"
) as handle:
    df_sum_temporal_features = pickle.load(handle)
    col = df_sum_temporal_features.columns
    df_sum_temporal_features.columns = [col[n] + " sum" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_sum_statistical_features.pickle"), "rb"
) as handle:
    df_sum_statistical_features = pickle.load(handle)
    col = df_sum_statistical_features.columns
    df_sum_statistical_features.columns = [col[n] + " sum" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_sum_spectral_features.pickle"), "rb"
) as handle:
    df_sum_spectral_features = pickle.load(handle)
    col = df_sum_spectral_features.columns
    df_sum_spectral_features.columns = [col[n] + " sum" for n in range(len(col))]

# df_max ##################################################################################
with open(
    os.path.join(cfg.pickle_dir, "df_max_temporal_features.pickle"), "rb"
) as handle:
    df_max_temporal_features = pickle.load(handle)
    col = df_max_temporal_features.columns
    df_max_temporal_features.columns = [col[n] + " max" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_max_statistical_features.pickle"), "rb"
) as handle:
    df_max_statistical_features = pickle.load(handle)
    col = df_max_statistical_features.columns
    df_max_statistical_features.columns = [col[n] + " max" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_max_spectral_features.pickle"), "rb"
) as handle:
    df_max_spectral_features = pickle.load(handle)
    col = df_max_spectral_features.columns
    df_max_spectral_features.columns = [col[n] + " max" for n in range(len(col))]

# df_xCe ##################################################################################
with open(
    os.path.join(cfg.pickle_dir, "df_xCe_temporal_features.pickle"), "rb"
) as handle:
    df_xCe_temporal_features = pickle.load(handle)
    col = df_xCe_temporal_features.columns
    df_xCe_temporal_features.columns = [col[n] + " xCe" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_xCe_statistical_features.pickle"), "rb"
) as handle:
    df_xCe_statistical_features = pickle.load(handle)
    col = df_xCe_statistical_features.columns
    df_xCe_statistical_features.columns = [col[n] + " xCe" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_xCe_spectral_features.pickle"), "rb"
) as handle:
    df_xCe_spectral_features = pickle.load(handle)
    col = df_xCe_spectral_features.columns
    df_xCe_spectral_features.columns = [col[n] + " xCe" for n in range(len(col))]

# df_yCe ##################################################################################
with open(
    os.path.join(cfg.pickle_dir, "df_yCe_temporal_features.pickle"), "rb"
) as handle:
    df_yCe_temporal_features = pickle.load(handle)
    col = df_yCe_temporal_features.columns
    df_yCe_temporal_features.columns = [col[n] + " yCe" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_yCe_statistical_features.pickle"), "rb"
) as handle:
    df_yCe_statistical_features = pickle.load(handle)
    col = df_yCe_statistical_features.columns
    df_yCe_statistical_features.columns = [col[n] + " yCe" for n in range(len(col))]

with open(
    os.path.join(cfg.pickle_dir, "df_yCe_spectral_features.pickle"), "rb"
) as handle:
    df_yCe_spectral_features = pickle.load(handle)
    col = df_yCe_spectral_features.columns
    df_yCe_spectral_features.columns = [col[n] + " yCe" for n in range(len(col))]

# df_AR ##################################################################################
with open(os.path.join(cfg.pickle_dir, "df_AR_features.pickle"), "rb") as handle:
    df_AR_features = pickle.load(handle)

# df_inter_stride #########################################################################
with open(os.path.join(cfg.pickle_dir, "df_inter_stride.pickle"), "rb") as handle:
    df_inter_stride = pickle.load(handle)
    df_inter_stride.rename("df_inter_stride", inplace=True)


# df_label ################################################################################
with open(os.path.join(cfg.pickle_dir, "df_label.pickle"), "rb") as handle:
    df_label = pickle.load(handle)


# df_vgg16 ################################################################################
with open(cfg.pickle_dir + "CNN_features.pickle", "rb") as handle:
    df_vgg16 = pickle.load(handle)

df_vgg16 = pd.DataFrame(df_vgg16, index=df_label.index)

# df_Mobnet ################################################################################
with open(cfg.pickle_dir + "CNN_features_MobileNet.pickle", "rb") as handle:
    df_Mobnet = pickle.load(handle)

df_Mobnet = pd.DataFrame(df_Mobnet, index=df_label.index)


df_temporal_features = pd.concat(
    [
        df_sum_temporal_features,
        df_max_temporal_features,
        df_xCe_temporal_features,
        df_yCe_temporal_features,
    ],
    axis=1,
)
df_statistical_features = pd.concat(
    [
        df_sum_statistical_features,
        df_max_statistical_features,
        df_xCe_statistical_features,
        df_yCe_statistical_features,
    ],
    axis=1,
)
df_spectral_features = pd.concat(
    [
        df_sum_spectral_features,
        df_max_spectral_features,
        df_xCe_spectral_features,
        df_yCe_spectral_features,
    ],
    axis=1,
)
df_all_features = pd.concat(
    [
        df_temporal_features,
        df_statistical_features,
        df_spectral_features,
        df_AR_features,
        df_inter_stride,
    ],
    axis=1,
)
# print(pd.DataFrame(labels_bi).head())
# print(df_all_features.head())


## deleting 30 last person as a unknown set
df_Mobnet = pd.concat([df_label, df_Mobnet], axis=1,)
indexNames = df_AR_features[df_Mobnet["ID"] > 5838].index
df_Mobnet.drop(indexNames, inplace=True)
df_Mobnet.drop(["ID"], axis=1, inplace=True)


df_vgg16 = pd.concat([df_label, df_vgg16], axis=1,)
indexNames = df_AR_features[df_vgg16["ID"] > 5838].index
df_vgg16.drop(indexNames, inplace=True)
df_vgg16.drop(["ID"], axis=1, inplace=True)


df_AR_features = pd.concat([df_label, df_AR_features], axis=1,)
indexNames = df_AR_features[df_AR_features["ID"] > 5838].index
df_AR_features.drop(indexNames, inplace=True)
df_AR_features.drop(["ID"], axis=1, inplace=True)


df_temporal_features = pd.concat([df_label, df_temporal_features], axis=1,)
indexNames = df_temporal_features[df_temporal_features["ID"] > 5838].index
df_temporal_features.drop(indexNames, inplace=True)
df_temporal_features.drop(["ID"], axis=1, inplace=True)


df_statistical_features = pd.concat([df_label, df_statistical_features], axis=1,)
indexNames = df_statistical_features[df_statistical_features["ID"] > 5838].index
df_statistical_features.drop(indexNames, inplace=True)
df_statistical_features.drop(["ID"], axis=1, inplace=True)


df_spectral_features = pd.concat([df_label, df_spectral_features], axis=1,)
indexNames = df_spectral_features[df_spectral_features["ID"] > 5838].index
df_spectral_features.drop(indexNames, inplace=True)
df_spectral_features.drop(["ID"], axis=1, inplace=True)

df_all_features = pd.concat([df_label, df_all_features], axis=1,)
indexNames = df_all_features[df_all_features["ID"] > 5838].index
df_all_features.drop(indexNames, inplace=True)
df_all_features.drop(["ID"], axis=1, inplace=True)


indexNames = df_label[df_label["ID"] > 5838].index
df_label.drop(indexNames, inplace=True)

print("[INFO] encoding labels...")
le = preprocessing.LabelEncoder()
labels_encoder = le.fit_transform(df_label)
# print(le.classes_)
# print(labels_encoder[100:300])
# from collections import Counter

# print(Counter(df_label.values.tolist()))
# 1 / 0
bi = preprocessing.MultiLabelBinarizer()
labels_bi = bi.fit_transform(df_label.values)
# print(pd.DataFrame(labels_bi).iloc[30:60, 0:30])
new_len = 20
fpr = np.zeros((1, new_len))
tpr = np.zeros((1, new_len))
thresholds = np.zeros((1, new_len))
print(labels_bi.shape)

print(Counter(df_label))  # equals to list(set(words))

for modelofperson in range(51):
    # print(labels_bi[:, model])

    print("[INFO] splitting the training and testing sets...")
    (trainData_AR, testData_AR, trainLabels_AR, testLabels_AR,) = train_test_split(
        df_AR_features,
        np.array(labels_bi[:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    (
        trainData_temporal,
        testData_temporal,
        trainLabels_temporal,
        testLabels_temporal,
    ) = train_test_split(
        df_temporal_features,
        np.array(labels_bi[:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    (
        trainData_statistical,
        testData_statistical,
        trainLabels_statistical,
        testLabels_statistical,
    ) = train_test_split(
        df_statistical_features,
        np.array(labels_bi[:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    (
        trainData_spectral,
        testData_spectral,
        trainLabels_spectral,
        testLabels_spectral,
    ) = train_test_split(
        df_spectral_features,
        np.array(labels_bi[:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    (trainData_all, testData_all, trainLabels_all, testLabels_all) = train_test_split(
        df_all_features,
        np.array(labels_bi[:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    (
        trainData_vgg16,
        testData_vgg16,
        trainLabels_vgg16,
        testLabels_vgg16,
    ) = train_test_split(
        df_vgg16,
        np.array(labels_bi[:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    (
        trainData_Mobnet,
        testData_Mobnet,
        trainLabels_Mobnet,
        testLabels_Mobnet,
    ) = train_test_split(
        df_Mobnet,
        np.array(labels_bi[:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    print("[INFO] Deleting High-correlated features...")
    if cfg.Highcorrelatedflag:
        corr_features = tsfel.correlated_features(trainData_AR)
        trainData_AR.drop(corr_features, axis=1, inplace=True)
        testData_AR.drop(corr_features, axis=1, inplace=True)

        corr_features = tsfel.correlated_features(trainData_temporal)
        trainData_temporal.drop(corr_features, axis=1, inplace=True)
        testData_temporal.drop(corr_features, axis=1, inplace=True)

        corr_features = tsfel.correlated_features(trainData_statistical)
        trainData_statistical.drop(corr_features, axis=1, inplace=True)
        testData_statistical.drop(corr_features, axis=1, inplace=True)

        corr_features = tsfel.correlated_features(trainData_spectral)
        trainData_spectral.drop(corr_features, axis=1, inplace=True)
        testData_spectral.drop(corr_features, axis=1, inplace=True)

        corr_features = tsfel.correlated_features(trainData_all)
        trainData_all.drop(corr_features, axis=1, inplace=True)
        testData_all.drop(corr_features, axis=1, inplace=True)

    print("[INFO] Deleting low variance features...")
    if cfg.VarianceThresholdflag:
        selector = VarianceThreshold()
        trainData_temporal = selector.fit_transform(trainData_temporal)
        testData_temporal = selector.transform(testData_temporal)

        trainData_statistical = selector.fit_transform(trainData_statistical)
        testData_statistical = selector.transform(testData_statistical)

        trainData_spectral = selector.fit_transform(trainData_spectral)
        testData_spectral = selector.transform(testData_spectral)

        trainData_all = selector.fit_transform(trainData_all)
        testData_all = selector.transform(testData_all)

        trainData_vgg16 = trainData_vgg16.values
        testData_vgg16 = testData_vgg16.values

        trainData_Mobnet = trainData_Mobnet.values
        testData_Mobnet = testData_Mobnet.values

    # print(type(trainData_all))
    # print(type(trainData_vgg16))
    # print(trainData_all.shape)
    # print(trainData_vgg16.shape)
    # 1 / 0
    print("[INFO] Standardization of features...")
    if cfg.transform == "standardization":
        scaler = preprocessing.StandardScaler()
        trainData_temporal = scaler.fit_transform(trainData_temporal)
        testData_temporal = scaler.transform(testData_temporal)

        trainData_statistical = scaler.fit_transform(trainData_statistical)
        testData_statistical = scaler.transform(testData_statistical)

        trainData_spectral = scaler.fit_transform(trainData_spectral)
        testData_spectral = scaler.transform(testData_spectral)

        trainData_all = scaler.fit_transform(trainData_all)
        testData_all = scaler.transform(testData_all)
    elif cfg.transform == "normalization":
        scaler = preprocessing.MinMaxScaler()
        trainData_temporal = scaler.fit_transform(trainData_temporal)
        testData_temporal = scaler.transform(testData_temporal)

        trainData_statistical = scaler.fit_transform(trainData_statistical)
        testData_statistical = scaler.transform(testData_statistical)

        trainData_spectral = scaler.fit_transform(trainData_spectral)
        testData_spectral = scaler.transform(testData_spectral)

        trainData_all = scaler.fit_transform(trainData_all)
        testData_all = scaler.transform(testData_all)
    elif cfg.transform == "none":
        pass

    # print(
    #     "[INFO] splitting the training dataset into {} folds\n\n".format(cfg.outer_n_splits)
    # )
    # cv_outer = StratifiedKFold(
    #     n_splits=cfg.outer_n_splits, shuffle=cfg.outer_shuffle, random_state=cfg.seed
    # )

    if cfg.features_name == "temporal":
        trainData = trainData_temporal
        trainLabels = trainLabels_temporal
        testData = testData_temporal
        testLabels = testLabels_temporal
    elif cfg.features_name == "statistical":
        trainData = trainData_statistical
        trainLabels = trainLabels_statistical
        testData = testData_statistical
        testLabels = testLabels_statistical
    elif cfg.features_name == "spectral":
        trainData = trainData_spectral
        trainLabels = trainLabels_spectral
        testData = testData_spectral
        testLabels = testLabels_spectral
    elif cfg.features_name == "all":
        trainData = trainData_all
        trainLabels = trainLabels_all
        testData = testData_all
        testLabels = testLabels_all
    elif cfg.features_name == "AR":
        trainData = trainData_AR
        trainLabels = trainLabels_AR
        testData = testData_AR
        testLabels = testLabels_AR
    elif cfg.features_name == "vgg16":
        trainData = trainData_vgg16
        trainLabels = trainLabels_vgg16
        testData = testData_vgg16
        testLabels = testLabels_vgg16
    elif cfg.features_name == "Mobnet":
        trainData = trainData_Mobnet
        trainLabels = trainLabels_Mobnet
        testData = testData_Mobnet
        testLabels = testLabels_Mobnet

    trainingData, evaluationData = trainData, testData
    trainingLabels, evaluationLabels = trainLabels, testLabels

    print("[INFO] training data shape : {}".format(trainingData.shape))
    print("[INFO] training labels shape : {}\n\n".format(trainingLabels.shape))

    print("[INFO] evaluation data shape : {}".format(evaluationData.shape))
    print("[INFO] evaluation labels shape : {}\n\n".format(evaluationLabels.shape))

    cv_inner = StratifiedKFold(
        n_splits=cfg.inner_n_splits, shuffle=cfg.inner_shuffle, random_state=cfg.seed
    )

    if cfg.classifier_name == "knn":
        # use kNN as the model
        print("[INFO] creating model...")
        model = KNeighborsClassifier()

        # sfs = SFS(
        #     estimator=model,
        #     forward=True,
        #     floating=False,
        #     scoring="accuracy",
        #     verbose=1,
        #     cv=cv_inner,
        #     n_jobs=cfg.Grid_n_jobs,
        # )
        # pipe = Pipeline([("sfs", sfs), ("model", model)])

        space = cfg.knnspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)
        best_model = result.best_estimator_

    elif cfg.classifier_name == "svm":
        # use SVM as the model
        print("[INFO] creating model...")
        model = svm.SVC()

        # sfs = SFS(
        #     estimator=model,
        #     forward=True,
        #     floating=False,
        #     scoring="accuracy",
        #     verbose=1,
        #     cv=cv_inner,
        #     n_jobs=cfg.Grid_n_jobs,
        # )
        # pipe = Pipeline([("sfs", sfs), ("model", model)])

        # define search space
        space = cfg.svmspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    elif cfg.classifier_name == "lda":
        # use LDA as the model
        print("[INFO] creating model...")
        model = LinearDiscriminantAnalysis()

        sfs = SFS(
            estimator=model,
            forward=True,
            floating=False,
            scoring="accuracy",
            verbose=1,
            cv=cv_inner,
            n_jobs=cfg.Grid_n_jobs,
        )
        # define search space
        pipe = Pipeline([("sfs", sfs), ("model", model)])

        space = cfg.ldaspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )

        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    elif cfg.classifier_name == "reg":
        # use logistic regression as the model
        print("[INFO] creating model...")
        model = LogisticRegression()

        # define search space
        space = cfg.regspace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    elif cfg.classifier_name == "tree":
        print("[INFO] creating model...")
        model = DecisionTreeClassifier(random_state=cfg.seed)

        # sfs = SFS(
        #     estimator=model,
        #     forward=True,
        #     floating=False,
        #     scoring="accuracy",
        #     verbose=1,
        #     cv=cv_inner,
        #     n_jobs=cfg.Grid_n_jobs,
        # )
        # pipe = Pipeline([("sfs", sfs), ("model", model)])
        # define search space
        space = cfg.treespace

        # define search
        search = GridSearchCV(
            model,
            space,
            scoring="accuracy",
            n_jobs=cfg.Grid_n_jobs,
            cv=cv_inner,
            refit=cfg.Grid_refit,
        )
        # execute search
        result = search.fit(trainingData, trainingLabels)

        best_model = result.best_estimator_

    else:
        print("[ERROR] could not find the classifier")

    print("[INFO] evaluating model...")
    f = open(os.path.join(cfg.output_dir, "result.txt"), "a")

    f.write("##################################################\n")
    f.write("################## the settings ##################\n")
    f.write("##################################################\n\n")
    f.write("model:                 {}\n".format(modelofperson))
    f.write("Grid_n_jobs:           {}\n".format(cfg.Grid_n_jobs))
    f.write("space:                 {}\n".format(space))
    f.write("inner_n_splits:        {}\n".format(cfg.inner_n_splits))
    # f.write("outer_n_splits (dis):  {}\n".format(cfg.outer_n_splits))
    f.write("df_all_features.shape: {}\n".format(df_all_features.shape))
    f.write("trainingData.shape:    {}\n".format(trainingData.shape))
    # f.write("evaluationData.shape:  {}\n".format(evaluationData.shape))
    f.write("testData.shape:        {}\n".format(testData.shape))
    f.write("test size:             {}\n\n\n".format(cfg.test_size))

    # write the accuracies of test set to file
    f.write("\n\n##################################################\n")
    f.write("###### summarize the estimated performance #######\n")
    f.write("####### of the best model on the test set ########\n")
    f.write("##################################################\n\n")

    pred = list()
    rank_1 = 0
    for (label, feature1) in zip(testLabels, testData):
        # predict the probability of each class label and
        # take the top-5 class labels

        predictions = best_model.predict_proba(np.atleast_2d(feature1))[0]
        predictions = np.argsort(predictions)[::-1][0]

        # rank-1 prediction increment
        # print(str(label)+" -----> "+str(predictions))
        pred.append(predictions)
        if label == predictions:
            rank_1 += 1

    # convert accuracies to percentages
    rank_1 = (rank_1 / float(len(testLabels))) * 100

    f.write("Rank-1: {:.2f}%\n".format(rank_1))

    f.write("best parameters are:\n {}\n\n".format(result.best_params_))

    # evaluate the model of test data
    preds = best_model.predict(testData)

    # write the classification report to file
    f.write("{}\n".format(classification_report(testLabels, preds)))

    # display the confusion matrix
    print("[INFO] confusion matrix")

    # plot the confusion matrix

    cm = confusion_matrix(testLabels, preds)
    # plt.figure()
    # sns.heatmap(cm, annot=True, cmap="Set2")
    # plt.savefig(os.path.join(cfg.fig_dir, cfg.result_name_file + ".png"))

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    # reflects the classifier’s ability to detect members of the positive class (pathological state)
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    # reflects the classifier’s ability to detect members of the negative class (normal state)
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    # reflects the frequency with which the classifier makes a mistake by classifying normal state as pathological
    FPR = FP / (FP + TN)
    # False negative rate
    # reflects the frequency with which the classifier makes a mistake by classifying pathological state as normal
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    f.write("\n###########################################################")
    f.write("\n###########################################################\n")
    f.write("False Positive (FP):\t\t\t\n {}\n\n".format(FP))
    f.write("False Negative (FN):\t\t\t\n {}\n\n".format(FN))
    f.write("True Positive (TP):\t\t\t\n {}\n\n".format(TP))
    f.write("True Negative (TN):\t\t\t\n {}\n\n".format(TN))
    f.write("True Positive Rate (TPR)(Recall):\t\t\n {} \n\n".format(TPR))
    f.write("True Negative Rate (TNR)(Specificity):\t\t\n {} \n\n".format(TNR))
    f.write("Positive Predictive Value (PPV)(Precision):\n {} \n\n".format(PPV))
    f.write("Negative Predictive Value (NPV):\n {} \n\n".format(NPV))
    f.write(
        "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):\t\t\n {} \n\n".format(
            FPR
        )
    )
    f.write(
        "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):\t\t\n {} \n\n".format(
            FNR
        )
    )
    f.write("False Discovery Rate (FDR):\t\t\n {} \n\n".format(FDR))
    f.write("Overall accuracy (ACC):\t\t\t\n {} \n\n".format(ACC))
    f.write("\n###########################################################")
    f.write("\n###########################################################")
    f.write("\nConfusion Matrix (CM): \n{}".format(cm))

    f.close()

    pred_proba = best_model.predict_proba(testData)

    fprt, tprt, thresholdst = roc_curve(testLabels, pred_proba[:, 1])
    delta = (len(fprt) - 1) / (new_len - 1)
    fprt = [interpolate(fprt, i * delta) for i in range(new_len)]
    fprt = np.expand_dims(fprt, axis=0)

    delta = (len(tprt) - 1) / (new_len - 1)
    tprt = [interpolate(tprt, i * delta) for i in range(new_len)]
    tprt = np.expand_dims(tprt, axis=0)

    delta = (len(thresholdst) - 1) / (new_len - 1)
    thresholdst = [interpolate(thresholdst, i * delta) for i in range(new_len)]
    thresholdst = np.expand_dims(thresholdst, axis=0)

    fpr = np.concatenate((fpr, fprt), axis=0)
    tpr = np.concatenate((tpr, tprt), axis=0)
    thresholds = np.concatenate((thresholds, thresholdst), axis=0)

    # showing best results along with the best parameters
    print("[INFO] Grid search best parameters: {}".format(result.best_params_))
    print("-------------------------------------------------------------------\n\n")
    print("[INFO] Accuracy Rank-1: {:.3f}%".format(rank_1))

    os.chdir(cfg.output_dir)
    if os.path.exists(cfg.output_dir + "/history_results/"):
        print("[INFO] The history folder exists")
    else:
        os.system("mkdir " + "history_results")

    os.chdir(cfg.output_dir + "/history_results/")
    if os.path.exists(cfg.output_dir + "/history_results/" + cfg.result_name_file):
        print("[INFO] The history folder exists")
    else:
        os.system("mkdir " + cfg.result_name_file)

    os.system(
        "mv -f "
        + os.path.join(cfg.output_dir, "result.txt")
        + " "
        + os.path.join(cfg.output_dir + "/history_results/")
        + cfg.result_name_file
        + "/"
        + str(modelofperson)
        + ".txt"
    )
# print(fpr)
# print(tpr)
fpr = fpr[1:, :]
tpr = tpr[1:, :]


plt.figure()
plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
# calculate roc curve for model
# fpr, tpr, _ = roc_curve(testy, pos_probs)
# # plot model roc curve
plt.plot(
    np.average(fpr, axis=0), np.average(tpr, axis=0), marker=".", label="average (LDA)"
)

# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# show the legend
plt.legend()
# show the plot
plt.savefig(os.path.join(cfg.fig_dir, "roc_curve " + cfg.result_name_file + ".png"))


print(auc(np.average(fpr, axis=0), np.average(tpr, axis=0)))

f = open(os.path.join(cfg.output_dir, "result.txt"), "a")
f.write("\n###########################################################")
f.write("\n###########################################################")
f.write("\nfpr (CM): \n{}".format(np.average(fpr, axis=0)))
f.write("\ntpr (CM): \n{}".format(np.average(tpr, axis=0)))
f.write(
    "\nauc (CM): \n{:2.3f}".format(
        auc(np.average(fpr, axis=0), np.average(tpr, axis=0))
    )
)

f.close()

os.system(
    "mv -f "
    + os.path.join(cfg.output_dir, "result.txt")
    + " "
    + os.path.join(cfg.output_dir + "/history_results/")
    + cfg.result_name_file
    + "/"
    + "00_fpr_tpr.txt"
)
plt.show()
# tpr = tpr[1:, :]
# print(auc(np.average(fpr, axis=0), np.average(tpr, axis=0)))
