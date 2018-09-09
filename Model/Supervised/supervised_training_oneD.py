# import default path
import os
import sys
sys.path.insert(0, '../../')

from Datamanager.OneD_Data_Creator      import OneD_Data_Loader
from Evaluation.Train_eval_1D_update    import VAE_evaluation
from Model.Modules.VAEs.VAE_Models_1D   import Basic_2Channel_VAE
from Model.Modules.Generators1D.Default_1D import VAE_RES, VAE_NO_RES
from Utils.cfg_utils import read_cfg

# - keras -
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Input
import keras
CUDA_VISIBLE_DEVICES=0

import numpy as np
import pickle

# - misc -
import sys


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = 0

    # generator
    config_name = 'vae_cfg_1D.ini'#sys.argv[1]
    vae_config = read_cfg(config_name, '../Configs')
    if vae_config['MODEL_TYPE'] != 'Vae':
        print('This is a routine to train Variational Autoencoders. Config of non VAE model was passed.')
        print('Exiting ...')
        exit()
    print(vae_config['DATASET'])
    Data_loader = OneD_Data_Loader(vae_config['DATASET'])
    vae_eval = VAE_evaluation(vae_config['MODEL_NAME'],Data_loader.config)

    # load Data ---------------------------------------------------------
    validation_split = vae_config['EVAL_SPLIT']

    xA_train = Data_loader.load_A()
    xB_train = Data_loader.load_B('B4')
    xA_test  = xA_train[0:int(xA_train.shape[0] * 0.1)]
    xB_test  = xB_train[0:int(xB_train.shape[0] * 0.1)]

    xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
    xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

    xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
    xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
    # /load Data --------------------------------------------------------

    optimizers = {'adam': keras.optimizers.adam}
    # Load Params
    id = ''
    vae_name        = vae_config['MODEL_NAME']
    epochs_id       = vae_config['Training']['EPOCHS_ID']
    epochs_norm     = vae_config['Training']['EPOCHS_NO']
    b_size_id       = vae_config['Training']['BATCH_SIZE_ID']
    b_size_no       = vae_config['Training']['BATCH_SIZE_NO']
    lr              = vae_config['VAE' + id]['LEARNING_RATE']
    lr_decay        = vae_config['VAE' + id]['LR_DEF']
    optimizer       = optimizers[vae_config['VAE' + id]['OPTIMIZER']](lr, lr_decay)
    filters         = vae_config['VAE' + id]['FILTERS']
    use_drop_out    = vae_config['VAE' + id]['USE_DROP_OUT']
    use_batch_normalisation = vae_config['VAE' + id]['USE_BATCH_NORM']
    trafo_layers            = vae_config['VAE' + id]['TRAFO_LAYERS']
    relu_param              = vae_config['VAE' + id]['RELU_PARAM']


    vae_res = VAE_NO_RES(vae_name, xA_train[0].shape, filters, relu_param, use_batch_normalisation, use_drop_out)
    # --- build gen ---

    vae_res.compile(optimizer=optimizer, loss=vae_config['VAE' + id]['IMAGE_LOSS'])#, metrics=['mean_absolute_error', 'val_mean_absolute_error'])

    # train gen, ID
    checkpointer0 = ModelCheckpoint(vae_eval.model_saves_dir + '/' + vae_name + '_weights.h5', verbose=1, save_best_only=True)
    vae_res.fit(xA_train, xA_train, validation_data=(xa_val, xa_val), shuffle=True, epochs=epochs_id, batch_size=b_size_id, callbacks=[checkpointer0])
    eval_score_id = vae_res.evaluate(xA_test, xA_test, batch_size=b_size_id)

    # train gen, no
    del vae_res
    vae_res = load_model(vae_eval.model_saves_dir + '/' + vae_name + '_weights.h5')
    checkpointer = ModelCheckpoint(vae_eval.model_saves_dir + '/' + vae_name + 'N_weights.h5', verbose=1, save_best_only=True)
    vae_res.fit(xB_train, xA_train, validation_data=(xb_val, xa_val), shuffle=True, epochs=epochs_norm, batch_size=b_size_no, callbacks=[checkpointer])
    eval_score_norm = vae_res.evaluate(xB_test, xA_test, batch_size=b_size_no)
    print(eval_score_norm)

    # evaluate the best model
    del vae_res
    vae_res = load_model(vae_eval.model_saves_dir + '/' + vae_name + 'N_weights.h5')
    eval_score_norm = vae_res.evaluate(xB_test, xA_test, batch_size=b_size_no)
    print(eval_score_norm)

    # make samples
    vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae_res.predict, vae_res.name, xB_test, xA_test, -1, obj='Eval_on_test')
    vae_eval.sample_best_model_output(xB_test, xA_test, vae_res.predict, vae_res.name)
    vae_eval.visualize_best_samples(vae_name)

    # ID
    del vae_res
    vae_res = load_model(vae_eval.model_saves_dir + '/' + vae_name + '_weights.h5')
    eval_score_norm = vae_res.evaluate(xA_test, xA_test, batch_size=b_size_id)
    print(eval_score_norm)

    # make samples
    vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae_res.predict, vae_res.name, xB_test, xA_test, -1,
                                                       obj='Eval_on_test_ID')
    vae_eval.sample_best_model_output(xA_test, xA_test, vae_res.predict, vae_res.name, obj='ID')
    vae_eval.visualize_best_samples(vae_name, obj='ID')

    del vae_res
    keras.backend.clear_session()






