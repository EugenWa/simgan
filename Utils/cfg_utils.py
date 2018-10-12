import os
import numpy as np
import configparser


def read_cfg(cfg_name, cfg_dir):
    cfg_path = cfg_dir + '/' +cfg_name
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)

    c = dict()

    # - Load general information -
    c['MODEL_NAME']     = cfg.get('general', 'modelname')
    c['SEED']           = cfg.getint('general', 'seed')                         # random seed
    c['DATASET']        = cfg.get('general', 'dataset')                         # dataset name
    c['DEBUG']          = cfg.getboolean('general', 'debug')                    # set version to release pr debug
    c['DESCRIPTION']    = cfg.get('general', 'description', fallback='')        # description of the cfg file
    c['EVAL_SPLIT']     = cfg.getfloat('general', 'eval_split', fallback=0.85)  # evaluation -/- training split
    c['MODEL_SAVE_DIR'] = cfg.get('general', 'model_save_directory')            # place to save the trained models
    c['METRIC']         = cfg.getint('general', 'metric', fallback=0)           # how to evaluate the result (error-type)
    c['FEATURE_CLASSIFY'] = cfg.getint('general', 'FEATURE_CLASSIFY', fallback=0)  # 0 dont train classification, 1 train both, 2 train only classificaion
    c['CLASSIF_LOSS']     = cfg.get('general', 'CLASSIF_LOSS', fallback='mae')

    # Get Model type
    c['MODEL_TYPE']     = cfg.get('general', 'model_type')
    if c['MODEL_TYPE'] == 'Gen':
        cfg_Gen(c, cfg)                                                         # routine to load Generator-data
    elif c['MODEL_TYPE'] == 'GAN':
        cfg_GAN(c, cfg)                                                         # routine to load GAN-data
    elif c['MODEL_TYPE'] == 'Vae':
        cfg_VAE(c, cfg)                                                         # routine to load VAE-data, only 1 vae
    else:
        print('No Model Specified')

    return c


def cfg_Gen(c, cfg):
    c['Generator'] = {}
    gen = c['Generator']
    gen['MODEL_ID']         = cfg.getint('model', 'model_ID', fallback=0)       # chooses the Model
    gen['LEARNING_RATE']    = cfg.getfloat('model', 'lr_rate', fallback=0.002)  # Learning rate
    gen['LR_DEF']           = cfg.getfloat('model', 'lr_def', fallback=0.5)     # Learning rate
    gen['OPTIMIZER']        = cfg.get('model', 'optimizer', fallback='adam')    # Optimizer to be used
    gen['LOSS']             = cfg.get('model', 'LOSS')                          # Losses to be optimized
    gen['LOSS_WEIGHTS']     = np.fromstring(cfg.get('model', 'LOSS_WEIGHTS'), dtype=np.float, sep=' ') .tolist()        # Weights of the different Losses

    c['Training'] = {}
    train = c['Training']
    train['EPOCHS_ID']      = cfg.getint('train', 'epochs_identity', fallback=0)# Epochs to train identity mapping
    train['EPOCHS_NO']      = cfg.getint('train', 'epochs_normal', fallback=0)  # Epochs for normal Training
    train['BATCH_SIZE_ID']  = cfg.getint('train', 'batch_size_id', fallback=28) # Batch sized to be used in id-map training
    train['BATCH_SIZE_NO']  = cfg.getint('train', 'batch_size_no', fallback=28) # Batch sized to be used in no-map training
    pass

def cfg_GAN(c, cfg):
    # Discriminator:
    c['DISC'] = {}
    disc = c['DISC']
    disc['PATCH_NUMBER_W']  = cfg.getint('disc', 'patch_amount_width', fallback=1)  # amount of patches in x-direction
    disc['PATCH_NUMBER_H']  = cfg.getint('disc', 'patch_amount_height', fallback=1) # amount of patches in y-direction
    disc['MODEL_ID']        = cfg.getint('disc', 'model_ID', fallback=0)            # chooses Discriminator-type
    disc['LEARNING_RATE']   = cfg.getfloat('disc', 'lr_rate', fallback=0.001)       # Learning rate
    disc['LR_DEF']          = cfg.getfloat('disc', 'lr_def', fallback=0.5)          # Learning rate
    disc['OPTIMIZER']       = cfg.get('disc', 'optimizer', fallback='adam')         # Optimizer to be used
    disc['LOSS']            = cfg.get('disc', 'LOSS', fallback='binary_crossentropy')                               # Losses to be optimized
    disc['RELU_PARAM']      = cfg.getfloat('disc', 'reluparam', fallback=0.3)           # Relu-alpha
    disc['USE_DROP_OUT']    = cfg.getboolean('disc', 'USE_DROP_OUT', fallback=False)    # use dropout?
    disc['USE_BATCH_NORM']  = cfg.getboolean('disc', 'USE_BATCH_NORM', fallback=True)   # use batch normalisation?

    # combined Model
    c['FULL_MODEL'] = {}
    full_mod = c['FULL_MODEL']
    full_mod['UNSUPERVISED']= cfg.getboolean('full_model', 'UNSUPERVISED', fallback=False)
    full_mod['VAE_NAME']    = cfg.get('full_model', 'vae_name', fallback='vae')
    full_mod['DISC_NAME']   = cfg.get('full_model', 'disc_name', fallback='discriminator')
    full_mod['DISC_LOSS']   = cfg.get('full_model', 'DISC_LOSS')
    full_mod['IMG_LOSS']    = cfg.get('full_model', 'IMAGE_LOSS')
    full_mod['FEATURE_LOSS']= cfg.get('full_model', 'FEATURE_LOSS')
    full_mod['LOSS_WEIGHTS'] = np.fromstring(cfg.get('full_model', 'LOSS_WEIGHTS', fallback='10 1'), dtype=np.float, sep=' ').tolist()
    full_mod['LOSS_WEIGHTS_NO'] = np.fromstring(cfg.get('full_model', 'LOSS_WEIGHTS_NO'), dtype=np.float, sep=' ').tolist()
    full_mod['LEARNING_RATE']   = cfg.getfloat('full_model', 'lr_rate', fallback=0.001)  # Learning rate
    full_mod['LR_DEF']          = cfg.getfloat('full_model', 'lr_def', fallback=0.5)  # Learning rate
    full_mod['OPTIMIZER']       = cfg.get('full_model', 'optimizer', fallback='adam')  # Optimizer to be used
    full_mod['LOSS']            = cfg.get('full_model', 'LOSS', fallback='mae')  # Losses to be optimized

    c['FULL_Training'] = {}
    ftrain = c['FULL_Training']
    ftrain['EPOCHS_ID']      = cfg.getint('full_training', 'epochs_identity', fallback=1)   # Epochs to train identity mapping
    ftrain['EPOCHS_NO']      = cfg.getint('full_training', 'epochs_normal', fallback=1)     # Epochs for normal Training
    ftrain['BATCH_SIZE_ID']  = cfg.getint('full_training', 'batch_size_id',fallback=28)     # Batch sized to be used in id-map training
    ftrain['BATCH_SIZE_NO']  = cfg.getint('full_training', 'batch_size_no', fallback=28)    # Batch sized to be used in no-map training
    ftrain['GENHIST_SIZE']   = cfg.getint('full_training', 'generator_history_size', fallback=3)    # length of generator history

    ftrain['DISC_TRAIN_ACTIVATION'] = cfg.getfloat('full_training', 'DISC_TRAIN_ACTIVATION', fallback=0.5)  # sets the probability of the disc training activation
    ftrain['DISC_TRAIN_MODE']   = cfg.get('full_training', 'DISC_TRAIN_MODE', fallback='MIN')  # Mode Probability or fixed amount
    ftrain['DISC_TRAIN_ApE']    = cfg.getint('full_training', 'DISC_TRAIN_ApE', fallback=1)   # amount of patches to be trained
    ftrain['DISC_TRAIN_RATIO']  = cfg.getint('full_training', 'DISC_TRAIN_RATIO', fallback=1)
    ftrain['PRE_TRAIN_D']       = cfg.getboolean('full_training', 'PRE_TRAIN_D', fallback=False)  # Pretrain disc


def cfg_VAE(c, cfg):
    c['VAE'] = {}
    vae = c['VAE']
    vae['MODEL_ID']         = cfg.getint('model', 'model_ID', fallback=0)       # chooses VAE-type
    vae['LEARNING_RATE']    = cfg.getfloat('model', 'lr_rate', fallback=0.001)  # Learning rate
    vae['LR_DEF']           = cfg.getfloat('model', 'lr_def', fallback=0.5)     # Learning rate
    vae['RELU_PARAM']       = cfg.getfloat('model', 'reluparam', fallback=0.3)  # Relu-alpha
    vae['OPTIMIZER']        = cfg.get('model', 'optimizer', fallback='adam')    # Optimizer to be used
    vae['DECODER_LOSS']     = cfg.get('model', 'DECODER_LOSS')                  # Losses to be optimized
    vae['IMAGE_LOSS']       = cfg.get('model', 'IMAGE_LOSS')                    # Losses to be optimized
    vae['FEATURE_LOSS']     = cfg.get('model', 'FEATURE_LOSS')                  # Losses to be optimized
    vae['TRAFO_LAYERS']     = cfg.getint('model', 'TRAFO_LAYERS', fallback=1)   # Losses to be optimized
    vae['USE_DROP_OUT']     = cfg.getboolean('model', 'USE_DROP_OUT', fallback=False)   # use dropout?
    vae['USE_BATCH_NORM']   = cfg.getboolean('model', 'USE_BATCH_NORM', fallback=True)# use batch normalisation?
    vae['NO_FIT_MODE']      = cfg.getint('model', 'NO_FIT_MODE', fallback=0)    # For two channel AE's only

    vae['LOSS_WEIGHTS']     = np.fromstring(cfg.get('model', 'LOSS_WEIGHTS'), dtype=np.float, sep=' ').tolist()             # Weights of the different Losses
    vae['FILTERS']          = np.fromstring(cfg.get('model', 'filters'), dtype=np.int, sep=' ') .tolist()
    vae['SKIPS']            = np.fromstring(cfg.get('model', 'SKIPS', fallback=''), dtype=np.int, sep=' ') .tolist()                     # layers to be residually connected

    c['Training'] = {}
    train = c['Training']
    train['LOAD_PRE_T_ID'] = cfg.getboolean('train', 'LOAD_PRE_T_ID', fallback=False)       # load the pretrained ID Model
    train['LOAD_PRE_T_NO'] = cfg.getboolean('train', 'LOAD_PRE_T_NO', fallback=False)       # load the pretrained NO Model

    train['EPOCHS_ID'] = cfg.getint('train', 'epochs_identity', fallback=1)     # Epochs to train identity mapping; 1 since most models do a load after
    train['EPOCHS_NO'] = cfg.getint('train', 'epochs_normal', fallback=1)       # Epochs for normal Training
    train['BATCH_SIZE_ID'] = cfg.getint('train', 'batch_size_id', fallback=28)  # Batch sized to be used in id-map training
    train['BATCH_SIZE_NO'] = cfg.getint('train', 'batch_size_no', fallback=28)  # Batch sized to be used in no-map training
