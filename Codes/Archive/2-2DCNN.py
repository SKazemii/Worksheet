# filter warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import preprocessing
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, Dense, GlobalAveragePooling2D

# import seaborn as sns
# import tsfel
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from PIL import Image

# Custom imports
import config as cfg
from scipy import signal
from skimage.transform import resize
import skimage

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

print("[INFO] importing libraries....")


# load time series data ##################################################################################
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


# creating CNN models  ##################################################################################
if cfg.model_name == "vgg16":
    base_model = VGG16(weights=cfg.weights)
    model = Model(input=base_model.input, output=base_model.get_layer("fc1").output)
    image_size = (224, 224)
elif cfg.model_name == "test":
    base_model = VGG16(weights=cfg.weights)
    model = Model(input=base_model.input, output=base_model.get_layer("fc1").output)
    image_size = (224, 224)
elif cfg.model_name == "vgg19":
    base_model = VGG19(weights=cfg.weights)
    base_model.summary()
    model = Model(input=base_model.input, output=base_model.get_layer("fc1").output)
    image_size = (224, 224)
elif cfg.model_name == "resnet50":
    base_model = ResNet50(
        input_tensor=Input(shape=(224, 224, 3)),
        include_top=cfg.include_top,
        weights=cfg.weights,
    )
    base_model.summary()
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)

    model = Model(input=base_model.input, outputs=predictions)
    model.summary()

    image_size = (224, 224)
elif cfg.model_name == "inceptionv3":
    base_model = InceptionV3(
        include_top=cfg.include_top,
        weights=cfg.weights,
        input_tensor=Input(shape=(299, 299, 3)),
    )
    # base_model.summary()
    # add a global spatial average pooling layer
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)

    model = Model(input=base_model.input, outputs=predictions)
    image_size = (299, 299)
elif cfg.model_name == "inceptionresnetv2":
    base_model = InceptionResNetV2(
        include_top=cfg.include_top,
        weights=cfg.weights,
        input_tensor=Input(shape=(299, 299, 3)),
    )
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)
    model = Model(input=base_model.input, output=predictions)
    image_size = (299, 299)
elif cfg.model_name == "mobilenet":
    base_model = MobileNet(
        include_top=cfg.include_top,
        weights=cfg.weights,
        input_tensor=Input(shape=(224, 224, 3)),
        input_shape=(224, 224, 3),
    )
    x = base_model.output
    predictions = GlobalAveragePooling2D()(x)
    model = Model(input=base_model.input, output=predictions)
    image_size = (224, 224)
elif cfg.model_name == "xception":
    base_model = Xception(weights=cfg.weights)
    model = Model(
        input=base_model.input, output=base_model.get_layer("avg_pool").output
    )
    model.summary()
    image_size = (299, 299)
else:
    base_model = None
print("[INFO] successfully loaded base model and model...")

df_sum = df_sum / df_sum.max().max()
df_max = df_max / df_max.max().max()
df_xCe = df_xCe / df_xCe.max().max()
# df_yCe = df_yCe.values[:, :, np.newaxis] / df_yCe.max().max()
# A = np.concatenate((df_sum, df_max, df_xCe), axis=2)

# variables to hold features and labels
features = []
labels = []

print(df_sum.shape)
print(df_sum.min().min())

# loop over all the labels in the folder
for i in range(df_sum.shape[1]):
    R = df_sum.T.iloc[i, :]
    G = df_max.T.iloc[i, :]
    B = df_xCe.T.iloc[i, :]

    widths = np.arange(1, 101)
    cwtmatrR = signal.cwt(R, signal.ricker, widths)
    cwtmatrG = signal.cwt(G, signal.ricker, widths)
    cwtmatrB = signal.cwt(B, signal.ricker, widths)
    rescale_coeffsR = resize(cwtmatrR, image_size, mode="constant")
    rescale_coeffsG = resize(cwtmatrG, image_size, mode="constant")
    rescale_coeffsB = resize(cwtmatrB, image_size, mode="constant")

    rescale_coeffsR = rescale_coeffsR[:, :, np.newaxis]
    rescale_coeffsG = rescale_coeffsG[:, :, np.newaxis]
    rescale_coeffsB = rescale_coeffsB[:, :, np.newaxis]

    RGB = np.concatenate((rescale_coeffsR, rescale_coeffsG, rescale_coeffsB), axis=2)
    # plt.imshow(
    #     RGB,
    #     extent=[0, 224, 1, 225],
    #     cmap="PRGn",
    #     aspect="auto",
    #     vmax=abs(RGB).max(),
    #     vmin=-abs(RGB).max(),
    # )
    # plt.show()

    RGB = np.expand_dims(RGB, axis=0)
    RGB = preprocess_input(RGB)
    feature = model.predict(RGB)
    flat = feature.flatten()
    features.append(flat)
    if i % 300 == 0:
        print("[INFO] completed images - " + str(i))


# get the shape of training labels
print("[STATUS] extracted features shape: {}".format(features[1].shape))
# save features and labels
with open(cfg.pickle_dir + "CNN_features_MobileNet.pickle", "wb") as handle:
    pickle.dump(np.array(features), handle, protocol=pickle.HIGHEST_PROTOCOL)


# save model and weights
# model_json = model.to_json()
# with open(cfg.pickle_dir + str(cfg.test_size * 100)[0:2] + ".json", "w") as json_file:
#     json_file.write(model_json)

# # save weights
# model.save_weights(cfg.model_path + str(cfg.test_size * 100)[0:2] + ".h5")
# print("[STATUS] saved model and weights to disk..")
# print("[STATUS] features and labels saved..")

# end time
# end = time.time()
# print(
#     "[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
# )


# print(df_sum.T.shape)
# x = df_max.T.iloc[1, :]
# # f, t, Zxx = signal.stft(x)
# # # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading="gouraud")
# # plt.pcolormesh(t, f, np.angle(Zxx), vmin=0, shading="gouraud")
# # plt.title("STFT Magnitude")
# # plt.ylabel("Frequency [Hz]")
# # plt.xlabel("Time [sec]")

# widths = np.arange(1, 31)
# cwtmatr = signal.cwt(x, signal.ricker, widths)
# plt.imshow(
#     cwtmatr,
#     extent=[0, 100, 1, 31],
#     cmap="PRGn",
#     aspect="auto",
#     vmax=abs(cwtmatr).max(),
#     vmin=-abs(cwtmatr).max(),
# )


# plt.figure()
# x = df_max.T.iloc[24, :]
# # f, t, Zxx = signal.stft(x)
# # # plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading="gouraud")
# # plt.pcolormesh(t, f, np.angle(Zxx), vmin=0, shading="gouraud")
# # plt.title("STFT Magnitude")
# # plt.ylabel("Frequency [Hz]")
# # plt.xlabel("Time [sec]")

# widths = np.arange(1, 31)
# cwtmatr = signal.cwt(x, signal.ricker, widths)
# plt.imshow(
#     cwtmatr,
#     extent=[0, 100, 1, 31],
#     cmap="PRGn",
#     aspect="auto",
#     vmax=abs(cwtmatr).max(),
#     vmin=-abs(cwtmatr).max(),
# )

# plt.show()
