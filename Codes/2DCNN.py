# filter warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import preprocessing, callbacks 
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten

# import seaborn as sns
# import tsfel
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys, logging
from PIL import Image
from pathlib import Path as Pathlb


# Custom imports
from scipy import signal
import seaborn as sns
# from skimage.transform import resize
# import skimage
# sns.set()

sys.path.insert(0, os.path.abspath(os.path.join('..')))
from MLPackage import config as cfg


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"



project_dir = os.getcwd()
log_path = os.path.join(project_dir, 'logs')

Pathlb(log_path).mkdir(parents=True, exist_ok=True)





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


logger.info("Importing libraries....")

if cfg.CNN["model_name"] == "vgg16":

    logger.info("Loading VGG16 model...")


    base_model = VGG16(
        weights=cfg.CNN["weights"], 
        include_top=cfg.CNN["include_top"])

    base_model.trainable = False

    input = tf.keras.Input(shape=cfg.CNN["image_size"], name="original_img")
    x = base_model(input)
    output = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = tf.keras.Model(input, output, name=cfg.CNN["model_name"])
    # model.summary() 
    tf.keras.utils.plot_model(model, to_file=cfg.CNN["model_name"]+".png", show_shapes=True)
   


elif cfg.CNN["model_name"] == "mobilenet":
    logger.info("Loading MobileNet model...")
    base_model = MobileNet(
        include_top = cfg.CNN["include_top"],
        weights = cfg.CNN["weights"],
        input_tensor=Input(shape=(224, 224, 3)),
        input_shape=(224, 224, 3),
    )
    # x = base_model.output
    # predictions = GlobalAveragePooling2D()(x)
    # model = Model(inputs=base_model.input, outputs=predictions)
    # image_size = (224, 224)

elif cfg.CNN["model_name"] == "resnet50":
    base_model = ResNet50(
        input_tensor=Input(shape=(224, 224, 3)),
        include_top = cfg.CNN["include_top"],
        weights = cfg.CNN["weights"],
    )
    # base_model.summary()
    # x = base_model.output
    # predictions = GlobalAveragePooling2D()(x)

    # model = Model(inputs=base_model.input, outputs=predictions)
    # model.trainable = False

    # image_size = (224, 224)
else:
    base_model = None
    logger.error("the model name is not correct!!!")




i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
x = tf.keras.applications.mobilenet.preprocess_input(x)
core = tf.keras.applications.MobileNet()
x = core(x)
model = tf.keras.Model(inputs=[i], outputs=[x])

image = tf.image.decode_png(tf.io.read_file('file.png'))
result = model(image)


logger.info("Successfully loaded base model and model...")
# base_model.summary()

saving_path = os.path.join(project_dir, 'Datasets', 'prefeatures.npy')
prefeatures = np.load(saving_path)
logger.info("prefeature shape: {}".format(prefeatures.shape))


# #CD, PTI, Tmax, Tmin, P50, P60, P70, P80, P90, P100
logger.info("batch_size: {}".format(cfg.CNN["batch_size"]))

for layer in base_model.layers:
    layer.trainable=False


x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(cfg.CNN["class_numbers"], activation = 'softmax')(x)

deep_model = Model(inputs=base_model.input, outputs=predictions)
deep_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )

checkpoint = [
        callbacks.ModelCheckpoint(
            cfg.CNN["saving_path"], save_best_only=True, monitor="val_loss"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        callbacks.EarlyStopping( monitor="val_loss", patience=50, verbose=1),
    ]    


# history = deep_model.fit(
#     X,
#     y,
#     batch_size = cfg.CNN["batch_size"],
#     callbacks = [checkpoint],
#     epochs = cfg.CNN["epochs"],
#     validation_split = cfg.CNN["validation_split"],
#     verbose = cfg.CNN["verbose"],
# )

# for i,layer in enumerate(deep_model.layers):
#     print(i,layer.name,layer.trainable)

# deep_model.summary()

sys.exit()

Deep_features = np.zeros((1, 2048))
for image in range(0, prefeatures.shape[0], cfg.CNN["batch_size"]):
    pic = prefeatures[image:image+cfg.CNN["batch_size"],:,:,0:3]
    pic = np.pad(pic, ((0, 0), (82, 82), (92, 92), (0, 0)), 'constant', constant_values = 0)

    pic = preprocess_input(pic)
    feature = deep_model.predict(pic)
    Deep_features = np.append(Deep_features, feature, axis=0)
    if image % 128 == 0:
        print("* ----- completed images: " + str(image))
    # print(Deep_features.shape)



    # plt.imshow(pic[-1,:,:,2])
    # plt.show()


Deep_features = Deep_features[1:, :]
print(Deep_features.shape)




# get the shape of training labels
print("Extracted features shape: {}".format(Deep_features.shape))
# save features and labels
with open(cfg.pickle_dir + "CNN_features_MobileNet.pickle", "wb") as handle:
    pickle.dump(np.array(Deep_features), handle, protocol=pickle.HIGHEST_PROTOCOL)