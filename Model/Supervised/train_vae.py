# import default path
import os
import sys
sys.path.insert(0, '../../')

from Datamanager.Data_loader        import Data_Loader
from Evaluation.Train_eval          import VAE_evaluation
from Model.Modules.VAEs.VAE_Models  import Basic_VAE
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
    config_name = sys.argv[1]
    vae_config = read_cfg(config_name, '../Configs')
    if vae_config['MODEL_TYPE'] != 'Vae':
        print('This is a routine to train Variational Autoencoders. Config of non VAE model was passed.')
        print('Exiting ...')
        exit()
    vae_eval = VAE_evaluation(vae_config['MODEL_NAME'])

    # load Data ---------------------------------------------------------
    D_loader = Data_Loader(vae_config['DATASET'])
    xA_train, xB_train = D_loader.Load_Data_Tensors('train', invert=True)
    xA_test, xB_test = D_loader.Load_Data_Tensors('test', invert=True)

    validation_split = vae_config['EVAL_SPLIT']

    xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
    xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

    xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
    xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
    # /load Data --------------------------------------------------------

    # Load Params
    vae_name        = vae_config['MODEL_NAME']
    epochs_id       = vae_config['Training']['EPOCHS_ID']
    epochs_norm     = vae_config['Training']['EPOCHS_NO']
    b_size_id       = vae_config['Training']['BATCH_SIZE_ID']
    b_size_no       = vae_config['Training']['BATCH_SIZE_NO']

    ####################
    vae = Basic_VAE(vae_name, xA_train[0].shape, vae_config)
    ####################


    # set equal shuffling
    np.random.seed(vae_config['SEED'])
    equal_shuffle = []
    for i in range(epochs_id + epochs_norm):            # shuffle all here in case the model uses random operations
        a = np.arange(xA_train.shape[0])                # which another doesnt -> rnd order caused by the seed will be permuted
        np.random.shuffle(a)
        equal_shuffle.append(a)

    # set training parameters
    train_shuffle = True


    for epoch in range(epochs_id + epochs_norm):
        if epoch < epochs_id:
            batch_size = b_size_id
        else:
            batch_size = b_size_no
        number_of_iterations = int(xA_train.shape[0] / batch_size)

        # shuffle set
        if train_shuffle:
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

        id_loss_batch = []
        no_loss_batch = []
        ges_loss_batch = []
        for batch_i in range(number_of_iterations):
            xA_batch = xA_train[batch_i*batch_size:(batch_i+1)*batch_size]
            xB_batch = xB_train[batch_i*batch_size:(batch_i+1)*batch_size]
            if epoch < epochs_id:
                loss_id, _, _ = vae.train_model_on_batch(xA_batch, xA_batch, True)
                print('Epoch ', epoch, '/', (epochs_id + epochs_norm), '; Batch ', batch_i,'/', number_of_iterations, ' ID-Loss: ', loss_id)
            else:
                loss_ges, loss_id, loss_no =vae.train_model_on_batch(xB_batch, xA_batch, False)
                print('Epoch ', epoch, '/', (epochs_id + epochs_norm), '; Batch ', batch_i, '/', number_of_iterations, ':')
                print('Ges-Loss:  ', loss_ges)
                print('ID-Loss:   ', loss_id)
                print('Map-Loss:  ', loss_no)
                no_loss_batch.append(loss_no)
                ges_loss_batch.append(loss_ges)

            id_loss_batch.append(loss_id)

        # calculate training loss
        if epoch >= epochs_id:
            vae_eval.no_loss_epoch.append([np.mean(no_loss_batch), np.std(no_loss_batch)])
            vae_eval.ges_loss_epoch.append([np.mean(ges_loss_batch), np.std(ges_loss_batch)])
        vae_eval.id_loss_epoch.append([np.mean(id_loss_batch), np.std(id_loss_batch)])


        # evaluate Model on Validation_set
        validation_loss_id_epoch = vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name, xa_val, xa_val, epoch, 'ID')
        vae_eval.valid_loss_id.append(validation_loss_id_epoch)
        # save best id model
        if vae_eval.q_save_model_ID(validation_loss_id_epoch, epoch):
            # safe model
            print('SAVING ID-Model')
            vae.save_model(vae_eval.model_saves_dir, obj='ID')

        if epoch >= epochs_id:
            validation_loss_no_epoch = vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name, xb_val, xa_val, epoch, 'NO')
            vae_eval.valid_loss_no.append(validation_loss_no_epoch)

            if vae_eval.q_save_model_NO(validation_loss_no_epoch, epoch):
                # safe model
                print('SAVING')
                vae.save_model(vae_eval.model_saves_dir)

    # save training history
    vae_eval.dump_training_history()



    print(' ---------------- TESTING ---------------- ')
    print()
    print('Test the best Identity-Model:')
    # --- Sample Best Model ---
    vae.load_Model(vae_eval.model_saves_dir, obj='ID')
    vae_eval.switch_best_epoch(True)                        # set best loss epoch to best id loss epoch
    # evaluate on test set
    Best_ID_Model_test_loss_id = vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name, xA_test, xA_test, -1, 'BEST_ID_id')
    Best_ID_Model_test_loss_no = vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name, xB_test, xA_test, -1, 'BEST_ID_no')
    print('Test loss ID: ', Best_ID_Model_test_loss_id)
    print('Test loss NO: ', Best_ID_Model_test_loss_no)
    vae_eval.sample_best_model_output(xA_test, xA_test, vae.predict_ID_img_only, vae.VAE_ID.name, 'BEST_ID_id')
    vae_eval.sample_best_model_output(xB_test, xA_test, vae.predict_NO_img_only, vae.VAE_NO.name, 'BEST_ID_no')

    print()
    print('Test the best Domainmapping-Model:')
    vae.load_Model(vae_eval.model_saves_dir)
    vae_eval.switch_best_epoch(False)                       # set best loss epoch to best no loss epoch
    Best_NO_Model_test_loss_id = vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_ID_img_only,vae.VAE_ID.name, xA_test, xA_test, -1, 'BEST_NO_id')
    Best_NO_Model_test_loss_no = vae_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_NO_img_only,vae.VAE_NO.name, xB_test, xA_test, -1, 'BEST_NO_no')
    print('Test loss ID: ', Best_NO_Model_test_loss_id)
    print('Test loss NO: ', Best_NO_Model_test_loss_no)

    vae_eval.sample_best_model_output(xA_test, xA_test, vae.predict_ID_img_only, vae.VAE_ID.name, 'BEST_NO_id' )
    vae_eval.sample_best_model_output(xB_test, xA_test, vae.predict_NO_img_only, vae.VAE_NO.name, 'BEST_NO_no')















