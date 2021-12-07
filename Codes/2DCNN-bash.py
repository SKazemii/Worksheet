# %% [markdown]
# # Importing Packages

# %%
# filter warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
import tensorflow as tf


from tensorflow.keras import preprocessing, callbacks 
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten

# import seaborn as sns
# import tsfel
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys, logging, h5py
from PIL import Image
from pathlib import Path as Pathlb


# Custom imports
from scipy import signal
import seaborn as sns


sns.set()

sys.path.insert(0, os.path.abspath(os.path.join('..')))
from MLPackage import Features as feat
from MLPackage import config as cfg


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# %% [markdown]
# # Setting Logger

# %%
project_dir = os.getcwd()
log_path = os.path.join(project_dir, 'logs')
temp_dir = os.path.join(project_dir, "temp")

Pathlb(log_path).mkdir(parents=True, exist_ok=True)



def create_logger(level):
    loggerName = "2DCNN_ipynb"
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


logger.info("Importing libraries....")

# %% [markdown]
# # Image Preprocessing and Loading

# %% [markdown]
# ## Loading Images

# %%
logger.info("[INFO] reading dataset....")
with h5py.File(cfg.configs["paths"]["stepscan_dataset"], "r") as hdf:
    barefoots = hdf.get("/barefoot/data")[:]
    metadata = hdf.get("/barefoot/metadata")[:]


data = barefoots.transpose(0,2,3,1)

logger.info(f"barefoots.shape: {data.shape}")
logger.info(f"metadata.shape: {metadata.shape}")
plt.imshow(data[1,:,:,:].sum(axis=2))


# %% [markdown]
# ## Extracting Image Features

# %%
features = list()
labels = list()

for label, sample in zip(metadata, data):
    # print(np.shape(sample))
    # print(np.max(sample))
    try:
        B = sample.sum(axis=1).sum(axis=0)
        A = np.trim_zeros(sample.sum(axis=1).sum(axis=0))
        aa = np.where(B == A[0])
        bb = np.where(B == A[-1])
        # print(aa[0][0])
        # print(bb[0][0])
        # print(np.trim_zeros(sample.sum(axis=1).sum(axis=0)))

        if aa[0][0]<bb[0][0]:
            features.append(feat.prefeatures(sample[10:70, 10:50, aa[0][0]:bb[0][0]]))
            labels.append(label)
        else:
            print(aa[0][0],bb[0][0])
            k=sample
            l=label
    except:
        continue
    # labels.append(label)
    # break
    # f = np.trim_zeros(sample.sum(axis=1).sum(axis=0),'f').shape[0]
    # b = np.trim_zeros(sample.sum(axis=1).sum(axis=0),'b').shape[0]
    # print(b, f,)

    # temp = np.zeros(sample[:,:,b:f].shape)
    # temp[sample[:,:,b:f] > 5] = 1
    # CD = np.sum(temp, axis=2)

    
        
    # # plt.imshow(sample[:,:,100])
    # print(np.max(sample[:,:,100]))
    # # break
    


logger.info(f"len prefeatures: {len(features)}")
logger.info(f"prefeatures.shape: {features[0].shape}")
logger.info(f"labels.shape: {labels[0].shape}")

np.save(os.path.join(temp_dir, 'prefeatures-SS.npy'), features)
np.save(os.path.join(temp_dir, 'metadata.npy'), labels)

# plt.imshow(CD)

# %% [markdown]
# ## Loading Image features

# %%
Loading_path = os.path.join(temp_dir, 'prefeatures-SS.npy')
prefeatures = np.load(Loading_path)
logger.info("prefeature shape: {}".format(prefeatures.shape))

Loading_path = os.path.join(temp_dir, 'metadata.npy')
metadata = np.load(Loading_path)
logger.info("prefeature shape: {}".format(metadata.shape))

# #CD, PTI, Tmax, Tmin, P50, P60, P70, P80, P90, P100
logger.info("batch_size: {}".format(cfg.configs["CNN"]["batch_size"]))

# %% [markdown]
# ## flatenning Images

# %%
images  = list()
for sample in prefeatures:
    sample = sample.transpose((2, 0, 1))

    total_image = sample[0,:,:]
    total_image1 = sample[5,:,:]

    for i in range(1,5):
        total_image = np.concatenate((total_image, sample[i,:,:]), axis=1)
        total_image1 = np.concatenate((total_image1, sample[i+5,:,:]), axis=1)




    total_image = np.concatenate((total_image, total_image1), axis=0)
    total_image = total_image[:,:, np.newaxis]
    total_image = np.concatenate((total_image, total_image, total_image), axis=2)

    images.append(total_image)

    # plt.figure(figsize=(20,20))
    # plt.imshow( total_image)
    # plt.show()

    # print(type(total_image))
    # print(total_image.dtype)
    # print(total_image.shape)



    # print(result)

    # break
images =np.array(images)
plt.imshow(images[55,...])

# %%

# indices = metadata[:,0]
# depth = len(np.unique(metadata[:,0]))
# one_hot_labels = tf.one_hot(indices, depth)

# %%
# CC=tf.keras.utils.to_categorical( metadata[:,0], num_classes=len(np.unique(metadata[:,0])))
# # one_hot_labels==CC
# len(np.unique(metadata[:,0]))
from sklearn import preprocessing as pre
indices = metadata[:,0]
le = pre.LabelEncoder()
le.fit(indices)
le.classes_
indices=le.transform(indices)
len(np.unique(indices))

# %% [markdown]
# # Making Base Model

# %%
try:
    logger.info(f"Loading { cfg.configs['CNN']['base_model'] } model...")
    base_model = eval("tf.keras.applications." + cfg.configs["CNN"]["base_model"] + "(weights=cfg.configs['CNN']['weights'], include_top=cfg.configs['CNN']['include_top'])")
    logger.info("Successfully loaded base model and model...")

except Exception as e: 
    
    base_model = None
    logger.error("The base model could NOT be loaded correctly!!!")
    print(e)


base_model.trainable = False

CNN_name = cfg.configs["CNN"]["base_model"].split(".")[0]

input = tf.keras.layers.Input(shape=cfg.configs["CNN"]["image_size"], dtype = tf.float64, name="original_img")
x = tf.cast(input, tf.float32)
x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
x = base_model(x)
x = tf.keras.layers.GlobalMaxPool2D()(x)
x = tf.keras.layers.Dense(256, activation='relu', name="last_dense")(x)
output = tf.keras.layers.Dense(cfg.configs['CNN']['class_numbers'], name="prediction")(x) # cfg.configs['CNN']['class_numbers']

# %% [markdown]
# # The CNN Model

# %%
model = tf.keras.models.Model(inputs=input, outputs=output, name=cfg.configs['CNN']['base_model'])

# Freeze the layers 
for layer in model.layers[-2:]:
    layer.trainable = True

# for i,layer in enumerate(model.layers):
#     print(i,layer.name,layer.trainable)

model.summary() 


tf.keras.utils.plot_model(model, to_file=cfg.configs['CNN']['base_model'] + ".png", show_shapes=True)
plt.show()


# %%
model.compile(
    optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
    )
time = int(timeit.timeit())*1_000_000
checkpoint = [
        callbacks.ModelCheckpoint(
            cfg.configs["CNN"]["saving_path"], save_best_only=True, monitor="val_loss"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        callbacks.EarlyStopping( monitor="val_loss", patience=50, verbose=1),
        callbacks.TensorBoard(log_dir=f'TensorBoard_logs/{time}')
        
    ]    


history = model.fit(
    images,
    indices,
    batch_size=cfg.configs["CNN"]["batch_size"],
    callbacks=[checkpoint],
    epochs= cfg.configs["CNN"]["epochs"],
    validation_split=cfg.configs["CNN"]["validation_split"],
    verbose=cfg.configs["CNN"]["verbose"],
)

# %%
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')




