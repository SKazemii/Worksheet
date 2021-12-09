 
# filter warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
import tensorflow as tf


from tensorflow.keras import preprocessing, callbacks 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten

# import seaborn as sns
# import tsfel
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys, logging, pprint, itertools, multiprocessing, copy
from PIL import Image
from pathlib import Path as Pathlb


# Custom imports
from scipy import signal
import seaborn as sns


sns.set()

sys.path.insert(0, os.path.abspath(os.path.join('..')))
from MLPackage import config as cfg
from MLPackage import Utilities as util


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"







# # Setting Logger


project_dir = cfg.configs["paths"]["project_dir"]
log_path = os.path.join(project_dir, "logs")
temp_dir = os.path.join(project_dir, "temp")

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




# # Making Base Model
def deep_features(configs):
    try:
        logger.info(f"Loading { configs['CNN']['base_model'] } model...")
        base_model = eval("tf.keras.applications." + configs["CNN"]["base_model"] + "(weights=configs['CNN']['weights'], include_top=configs['CNN']['include_top'])")
        logger.info("Successfully loaded base model and model...")

    except Exception as e: 
        base_model = None
        logger.error("The base model could NOT be loaded correctly!!!")
        print(e)


    base_model.trainable = False

    CNN_name = configs['CNN']["base_model"].split(".")[0]
    logger.info("MaduleName: {}\n".format(CNN_name))
    
    
    input = tf.keras.layers.Input(shape=configs['CNN']["image_size"], dtype = tf.float64, name="original_img")
    x = tf.cast(input, tf.float32)
    x = eval("tf.keras.applications." + CNN_name + ".preprocess_input(x)")
    x = base_model(x)
    output = tf.keras.layers.GlobalMaxPool2D()(x)



    model = tf.keras.Model(input, output, name=CNN_name)
    tf.keras.utils.plot_model(model, to_file=CNN_name + ".png", show_shapes=True)

    if configs['CNN']["verbose"]==True:
        model.summary() 




    # AUTOTUNE = tf.data.AUTOTUNE

    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



    # # Image Preprocessing and Loading
    # ## Loading Images

    Loading_path = os.path.join(project_dir, 'Datasets', 'prefeatures.npy')
    prefeatures = np.load(Loading_path)
    logger.info("prefeature shape: {}".format(prefeatures.shape))


    maxvalues = [np.max(prefeatures[...,ind]) for ind in range(len(cfg.image_feature_name))]

    for i in range(len(cfg.image_feature_name)):
        prefeatures[..., i] = prefeatures[..., i]/maxvalues[i]


    meta_path = os.path.join(project_dir, 'Datasets', 'metadatalist.npy')
    metadata = np.load(meta_path)
    logger.info("metadata shape: {}".format(metadata.shape))


    # #CD, PTI, Tmax, Tmin, P50, P60, P70, P80, P90, P100
    logger.info("batch_size: {}".format(configs['CNN']["batch_size"]))


    # # Extracting features



    Deep_features = np.zeros((1, model.layers[-1].output_shape[1]))

    train_ds = tf.data.Dataset.from_tensor_slices((prefeatures, metadata[:,0]))
    train_ds = train_ds.batch(configs['CNN']["batch_size"])





    for image_batch, labels_batch in train_ds:

        if configs['CNN']["image_feature"]=="tile":
            tile_images = util.tile(image_batch)
            feature = model(tile_images)
            Deep_features = np.append(Deep_features, feature, axis=0)
            if (Deep_features.shape[0]-1) % 256 == 0:
                logger.info(f" ->>> ({os.getpid()}) completed images: " + str(Deep_features.shape[0]))
        
        
        else:
            image_feature_name = dict(zip(cfg.image_feature_name, range(len(cfg.image_feature_name))))
            ind = image_feature_name[configs['CNN']["image_feature"]]
            
            images = image_batch[...,ind]
            images = images[...,tf.newaxis]
            images = np.concatenate((images, images, images), axis=-1)

            feature = model(images)
            Deep_features = np.append(Deep_features, feature, axis=0)
            # print(Deep_features.shape[0]-1)
            if (Deep_features.shape[0]-1) % 256 == 0:
                logger.info(f" ->>> ({os.getpid()}) completed images: " + str(Deep_features.shape[0]))


    Deep_features = Deep_features[1:, :]
    logger.info(f"Deep features shape: {Deep_features.shape}")



    
    # # Saving Featurs


    time = int(timeit.default_timer() * 1_000_000)

    file_name =  CNN_name + '_' + configs['CNN']["image_feature"] +'_features.xlsx'
    saving_path = os.path.join(temp_dir, file_name)
    columnsName = [CNN_name+"_"+str(i) for i in range(Deep_features.shape[1])]  + cfg.label
    Deep_features = np.concatenate((Deep_features, metadata[:Deep_features.shape[0], 0:2]), axis=1)

    try:
        pd.DataFrame(Deep_features, columns=columnsName).to_excel(saving_path)
    except Exception as e:
        print(e)
        pd.DataFrame(Deep_features, columns=columnsName).to_excel(os.path.join(temp_dir, file_name+str(time)+'.xlsx'))


def main():


    p  = [ "vgg16.VGG16", "resnet50.ResNet50", "efficientnet.EfficientNetB0", "mobilenet.MobileNet"]
    p1 = ["CD", "PTI", "Tmax", "Tmin", "P50", "P60", "P70", "P80", "P90", "P100", "tile"]
    space = list(itertools.product(p,p1))
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    ncpus = 4

    pool = multiprocessing.Pool(processes=ncpus)
    logger.info(f"CPU count: {ncpus}")
    for parameters in space:
        configs = copy.deepcopy(cfg.configs)
        configs["CNN"]["base_model"] = parameters[0]
        configs["CNN"]["image_feature"] = parameters[1]
        if parameters[1]=="tile":
            configs["CNN"]["image_size"] =  (120, 200, 3)
        else:
            configs["CNN"]["image_size"] =  (60, 40, 3)
        # pprint.pprint(configs)
        # breakpoint()
        pool.apply_async(deep_features, args=(configs,))
        # deep_features(configs)
        
    pool.close()
    pool.join()



    logger.info("Done!!!")



if __name__ == "__main__":
    logger.info("Starting !!!")
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    logger.info("Done ({:2.2f} process time)!!!\n\n\n".format(toc-tic))



