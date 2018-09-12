# import default path
import os
import sys
sys.path.insert(0, '../../')

from Datamanager.OneD_Data_Creator                      import OneD_Data_Loader
from Evaluation.Train_eval_1D_update                    import GAN_evaluation
from Model.Modules.Generators1D.Default_1D              import VAE_RES
from Model.Modules.Discriminators1D.Disc_1D             import discriminator_build_4conv_oneD
from Utils.cfg_utils import read_cfg

# - keras -
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
from keras.layers import Input, Lambda, Average
import keras
CUDA_VISIBLE_DEVICES=0
import numpy as np
import pickle

# - misc -
import sys


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = 0

    # generator
    config_name      = 'gan_cfg_oneD_1.ini'#sys.argv[1]  #'gan_cfg.ini'#sys.argv[1]
    degraded_dataset = 'B4'#sys.argv[2]
    gan_config = read_cfg(config_name, '../Configs')
    if gan_config['MODEL_TYPE'] != 'GAN':
        print('This is a routine to train GANs. Config of non GAN model was passed.')
        print('Exiting ...')
        exit()
    Data_loader = OneD_Data_Loader(gan_config['DATASET'])
    gan_eval = GAN_evaluation(gan_config['MODEL_NAME'], Data_loader.config)

    # load Data ---------------------------------------------------------
    validation_split = gan_config['EVAL_SPLIT']

    xA_train = Data_loader.load_A()
    xB_train = Data_loader.load_B(degraded_dataset)


    vae = load_model(gan_eval.model_saves_dir + '/EvalM')
    #discriminator = load_model(gan_eval.model_saves_dir + '/discriminator_LAST.h5')

    VAE_Transformed = vae.predict(xB_train)


    xA_test = xA_train[0:int(xA_train.shape[0] * 0.1)]
    xB_test = xB_train[0:int(xB_train.shape[0] * 0.1)]
    VAE_Transformed_test = VAE_Transformed[0:int(xB_train.shape[0] * 0.1)]

    xA_train = xA_train[int(xA_train.shape[0] * 0.1):]
    xB_train = xB_train[int(xB_train.shape[0] * 0.1):]
    VAE_Transformed_train = VAE_Transformed[int(xB_train.shape[0] * 0.1):]

    xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
    xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]
    trafo_val = VAE_Transformed_test[0:int(VAE_Transformed_test.shape[0] * validation_split)]

    xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
    xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
    VAE_Transformed_test = VAE_Transformed_test[int(VAE_Transformed_test.shape[0] * validation_split):]
    # /load Data --------------------------------------------------------

    train_set   = np.zeros((xA_train.shape[0] + VAE_Transformed_train.shape[0],) + xA_train[0].shape)
    val_set     = np.zeros((xa_val.shape[0] + trafo_val.shape[0],) + xa_val[0].shape)
    test_set    = np.zeros((xA_test.shape[0] + VAE_Transformed_test.shape[0],) + xA_test[0].shape)

    train_set[0:xA_train.shape[0]]  = xA_train
    train_set[xB_train.shape[0]:]   = VAE_Transformed_train

    val_set[0:xa_val.shape[0]] = xa_val
    val_set[xa_val.shape[0]:]  = trafo_val

    test_set[0:xA_test.shape[0]] = xA_test
    test_set[xA_test.shape[0]:]  = VAE_Transformed_test

    # generate

    labels_train    = np.zeros((xB_train.shape[0] + VAE_Transformed_train.shape[0],) )
    labels_train[0:xA_train.shape[0]] = np.ones((xA_train.shape[0],))

    labels_val      = np.zeros((xb_val.shape[0] + trafo_val.shape[0],))
    labels_val[0:xb_val.shape[0]] = np.ones((xb_val.shape[0],))

    labels_test     = np.zeros((xA_test.shape[0] + VAE_Transformed_test.shape[0],))
    labels_test[0:xA_test.shape[0]] = np.ones((xA_test.shape[0],))


    # discriminator
    patch_lenght_width = 1#gan_config['DISC']['PATCH_NUMBER_W']
    patch_width = int(xA_train[0].shape[0] / patch_lenght_width)

    disc_input = Input(shape=(patch_width, xA_train[0].shape[1]))
    # --- build_model ---
    disc_output = discriminator_build_4conv_oneD(disc_input, 0.3, True, True)

    optimizers = {'adam': keras.optimizers.adam}

    lr = gan_config['DISC']['LEARNING_RATE']
    lr_decay = gan_config['DISC']['LR_DEF']
    disc_optimizer = optimizers[gan_config['DISC']['OPTIMIZER']](lr, lr_decay)
    discriminator = Model(disc_input, disc_output, name=gan_config['FULL_MODEL']['DISC_NAME'])
    discriminator.compile(optimizer=disc_optimizer, loss=gan_config['DISC']['LOSS'], metrics=['accuracy'])


    # fit
    Modelcheckpointer = ModelCheckpoint(gan_eval.model_saves_dir + '/DISC_EXPERIMENT', verbose=1, save_best_only=True)
    history = discriminator.fit(train_set, labels_train, validation_data=(val_set, labels_val), shuffle=True, batch_size=128, epochs=10, callbacks=[Modelcheckpointer])
    eval_score = discriminator.evaluate(test_set, labels_test, batch_size=128)
    print('Eval score Last: ', eval_score)

    discriminator = load_model(gan_eval.model_saves_dir + '/DISC_EXPERIMENT')
    eval_score = discriminator.evaluate(test_set, labels_test, batch_size=128)
    print('Eval score Best: ', eval_score)


