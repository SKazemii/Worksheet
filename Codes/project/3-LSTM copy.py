# lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical

import datetime
import os, pickle
import matplotlib.pyplot as plt
import config as cfg
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)


print("[INFO] importing pickles files....")
with open(os.path.join(cfg.pickle_dir, "df_sum.pickle"), "rb") as handle:
    df_sum = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_max.pickle"), "rb") as handle:
    df_max = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_xCe.pickle"), "rb") as handle:
    df_xCe = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_yCe.pickle"), "rb") as handle:
    df_yCe = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_label.pickle"), "rb") as handle:
    df_label = pickle.load(handle)

df_sum = df_sum / df_sum.max().max()
df_max = df_max / df_max.max().max()
df_xCe = df_xCe / df_xCe.max().max()
df_yCe = df_yCe / df_yCe.max().max()

df_sum = pd.concat([df_label, df_sum.T], axis=1,)
indexNames = df_sum[df_sum["ID"] > 5838].index
df_sum.drop(indexNames, inplace=True)
df_sum.drop(["ID"], axis=1, inplace=True)

df_max = pd.concat([df_label, df_max.T], axis=1,)
indexNames = df_max[df_max["ID"] > 5838].index
df_max.drop(indexNames, inplace=True)
df_max.drop(["ID"], axis=1, inplace=True)

df_xCe = pd.concat([df_label, df_xCe.T], axis=1,)
indexNames = df_xCe[df_xCe["ID"] > 5838].index
df_xCe.drop(indexNames, inplace=True)
df_xCe.drop(["ID"], axis=1, inplace=True)

df_yCe = pd.concat([df_label, df_yCe.T], axis=1,)
indexNames = df_yCe[df_yCe["ID"] > 5838].index
df_yCe.drop(indexNames, inplace=True)
df_yCe.drop(["ID"], axis=1, inplace=True)

indexNames = df_label[df_label["ID"] > 5838].index
df_label.drop(indexNames, inplace=True)

R = df_sum.values[:, :, np.newaxis]
G = df_max.values[:, :, np.newaxis]
B = df_xCe.values[:, :, np.newaxis]
D = df_yCe.values[:, :, np.newaxis]

Signals = np.concatenate((R, G, B, D), axis=2)


print("[INFO] encoding labels...")
bi = preprocessing.MultiLabelBinarizer()
labels_bi = bi.fit_transform(df_label.values)

new_len = 10
fpr = np.zeros((1, new_len))
tpr = np.zeros((1, new_len))
thresholds = np.zeros((1, new_len))

print(Signals.shape)
print(labels_bi.shape)
print(Signals.min().min())
print(Signals.max().max())


def interpolate(inp, fi):
    i, f = (
        int(fi // 1),
        fi % 1,
    )  # Split floating-point index into whole & fractional parts.
    j = i + 1 if round(f, 4) > 0 else i  # Avoid index error.
    return (1 - f) * inp[i] + f * inp[j]


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):

    global fpr
    global tpr
    global thresholds

    verbose, epochs, batch_size = 1, 5, 64
    n_timesteps, n_features, n_outputs = (
        trainX.shape[1],
        trainX.shape[2],
        1,  # trainy.shape[1],
    )

    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(n_outputs, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )  # (loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    pred_proba = model.predict_proba(testData)
    print(pred_proba)
    print(pred_proba.shape)
    fprt, tprt, thresholdst = roc_curve(testy, pred_proba)
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

    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print("Accuracy: %.3f%% (+/-%.3f)" % (m, s))


for modelofperson in range(1):
    print("[INFO] splitting the training and testing sets...")
    (trainData, testData, trainLabels, testLabels,) = train_test_split(
        Signals,
        np.array(labels_bi[:, modelofperson]),  # [:, modelofperson]),
        stratify=np.array(labels_bi[:, modelofperson]),  # [:, modelofperson]),
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )
    print(trainLabels.shape)
    scores = list()
    for r in range(1):
        score = evaluate_model(trainData, trainLabels, testData, testLabels)
        score = score * 100.0
        print(">#%d: %.3f" % (r + 1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)


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

result_name_file = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + "-LSTM"


os.chdir(cfg.output_dir + "/history_results/")
if os.path.exists(cfg.output_dir + "/history_results/" + result_name_file):
    print("[INFO] The history folder exists")
else:
    os.system("mkdir " + result_name_file)

os.system(
    "mv -f "
    + os.path.join(cfg.output_dir, "result.txt")
    + " "
    + os.path.join(cfg.output_dir + "/history_results/")
    + result_name_file
    + "/"
    + "00_fpr_tpr.txt"
)
plt.show()
