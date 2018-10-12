# import default path
import sys
sys.path.insert(0, '../')

from Datamanager.OneD_Data_Creator                      import OneD_Data_Loader
from Evaluation.Train_eval_1D                           import GAN_evaluation, VAE_evaluation
from Evaluation.Evaluation_Visualisation                import GAN_visualisation, VAE_visualisation

from Model.Modules.GANs.Basic_AE_Gan import Basic_AE_Gan
from Model.Modules.AutoEncoders.Basic_1D_AutoEncoders import Basic_nRES_AE
from Model.Modules.AutoEncoders.TwoChannel_1D_AutoEncoders import Two_Channel_Auto_Encoder
from Model.Modules.AutoEncoders.Basic_1D_Residual_AutoEncoders import Basic_nRES_ResidualAE
from Utils.cfg_utils import read_cfg
import datetime

# - keras -
import keras
import numpy as np
import random
import pickle
import datetime

import seaborn

def finalize_VAE_train_onlyD(vae_args):
    vae_config_name = vae_args[0]
    degraded_dataset = vae_args[1]
    ae_config = read_cfg(vae_config_name, '../Model/Configs')
    if ae_config['MODEL_TYPE'] != 'Vae':
        print('This is a routine to train (Variational) Auto Encoders. Config of non (V)-AE model was passed.')
        print('Exiting ...')
        exit()
    print(ae_config['MODEL_NAME'])
    print('Dataset to be operated on: ', ae_config['DATASET'])
    Data_loader = OneD_Data_Loader(ae_config['DATASET'])
    ae_eval = VAE_evaluation('../Evaluation Results/', ae_config['MODEL_NAME'], '../Model/Configs' + '/' + vae_config_name, Data_loader.config)

    try:
    # get the best Epoch and Visualize the best Images
        with open(ae_eval.training_history_dir + '/Hist_ID', "rb") as fp:
            history_id = pickle.load(fp)

        min_id_value = min(history_id['val_loss'])
        min_id_epoch = history_id['val_loss'].index(min_id_value)
        print('Best Validation Epoch: ', min_id_epoch, '/', len(history_id['val_loss']), ' with value: ', min_id_value)
        print('--------------------------------------------------------------------------')
    except Exception:
        print('NO id hist')

    try:
        # get the best Epoch and Visualize the best Images
        with open(ae_eval.training_history_dir + '/Hist_NO', "rb") as fp:
            history_no = pickle.load(fp)

        min_no_value = min(history_no['val_loss'])
        min_no_epoch = history_no['val_loss'].index(min_no_value)
        print('Best Validation Epoch: ', min_no_epoch, '/', len(history_no['val_loss']), ' with value: ', min_no_value)
        print('--------------------------------------------------------------------------')
    except Exception:
        print('NO id hist')





def finalize_VAE_train(vae_args):
    vae_config_name  = vae_args[0]
    degraded_dataset = vae_args[1]
    ae_config = read_cfg(vae_config_name, '../Model/Configs')
    if ae_config['MODEL_TYPE'] != 'Vae':
        print('This is a routine to train (Variational) Auto Encoders. Config of non (V)-AE model was passed.')
        print('Exiting ...')
        exit()
    print(ae_config['MODEL_NAME'])
    print('Dataset to be operated on: ', ae_config['DATASET'])
    Data_loader = OneD_Data_Loader(ae_config['DATASET'])
    ae_eval = VAE_evaluation('../Evaluation Results/', ae_config['MODEL_NAME'], '../Model/Configs' + '/' + vae_config_name, Data_loader.config)
    ae_vis  = VAE_visualisation('../Evaluation Results/', ae_config['MODEL_NAME'], Data_loader.config, 0)

    # load Data ----------------------------------------------------------
    validation_split = ae_config['EVAL_SPLIT']
    classify_mode    = ae_config['FEATURE_CLASSIFY']
    '''
    if classify_mode is 1:
        xA_train_img_only, xA_target, xB_train, xa_val_img_only, xa_val_target, xb_val, xA_test_img_only, xA_test_target, xB_test = Data_loader.Load_Data_Tensors_WFeat(degraded_dataset, validation_split, 1)  # normal Data and features
    elif classify_mode is 2:
        xA_train_img_only, xA_target, xB_train, xa_val_img_only, xa_val_target, xb_val, xA_test_img_only, xA_test_target, xB_test = Data_loader.Load_Data_Tensors_WFeat(degraded_dataset, validation_split, 2)
    else:
        xA_train_img_only, xA_target, xB_train, xa_val_img_only, xa_val_target, xb_val, xA_test_img_only, xA_test_target, xB_test = Data_loader.Load_Data_Tensors_WFeat(degraded_dataset, validation_split, 0)
    # /load Data ---------------------------------------------------------
    '''

    xA_train, xB_train, xa_val, xb_val, xA_test, xB_test = Data_loader.Load_Data_Tensors(degraded_dataset, validation_split)
    # load Models
    ae_name                     = ae_config['MODEL_NAME']
    model_identification_num    = ae_config['VAE']['MODEL_ID']
    epochs_id = ae_config['Training']['EPOCHS_ID']
    epochs_norm = ae_config['Training']['EPOCHS_NO']
    b_size_id = ae_config['Training']['BATCH_SIZE_ID']
    b_size_no = ae_config['Training']['BATCH_SIZE_NO']
    Models = [Basic_nRES_AE, Basic_nRES_ResidualAE, Two_Channel_Auto_Encoder]
    AE_Model = Models[model_identification_num](ae_name, xA_train[0].shape, ae_config, ae_eval)




    if epochs_id > 0:
        AE_Model.load_pretrained_model('ID')        # load in the trained Model

        # Evaluate on test set
        eval_score_id_id = AE_Model.Call().evaluate(xA_test, xA_test, b_size_id)
        print('Evaluation Score on Test-set, ID-Model, ID->ID: ', eval_score_id_id)
        print('------------------------------------------------------------------')

        # sample Model
        ae_eval.sample_best_model_output(xA_test, xA_test, AE_Model.Call().predict, AE_Model.name, 'id2id_', 'ID Model')
        # visualize samples
        ae_vis.visualize_best_samples(AE_Model.name, 'id2id_', 'ID Model')

        # get the best Epoch and Visualize the best Images
        with open(ae_eval.training_history_dir + '/Hist_ID', "rb") as fp:
            history_id = pickle.load(fp)

        min_id_value = min(history_id['val_loss'])
        min_id_epoch = history_id['val_loss'].index(min_id_value)
        print('Best Validation Epoch: ', min_id_epoch, ' with value: ', min_id_value)

        ae_eval.evaluate_model_on_testdata_chunk_gen_only(AE_Model.Call().predict, AE_Model.name, xa_val, xa_val, -1, '', 'ID Model')
        ae_vis.visualize_chunk_data_test(AE_Model.name, xa_val, xa_val, -1, '', 'ID Model')

        print('--------------------------------------------------------------------------')

    # train gen, NO
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if epochs_norm > 0:
        AE_Model.load_pretrained_model('NO')        # load in the trained Model

        # Evaluate on test set
        eval_score_no_id = AE_Model.Call().evaluate(xA_test, xA_test, b_size_no)
        eval_score_no_no = AE_Model.Call().evaluate(xB_test, xA_test, b_size_no)
        print('Evaluation Score on Test-set, NO-Model, ID->ID: ', eval_score_no_id)
        print('Evaluation Score on Test-set, NO-Model, NO->ID: ', eval_score_no_no)
        print('------------------------------------------------------------------')

        # sample Model
        ae_eval.sample_best_model_output(xA_test, xA_test, AE_Model.Call().predict, AE_Model.name, 'id2id_', 'NO Model')
        ae_eval.sample_best_model_output(xB_test, xA_test, AE_Model.Call().predict, AE_Model.name, 'no2id_', 'NO Model')
        # visualize samples
        ae_vis.visualize_best_samples(AE_Model.name, 'id2id_', 'NO Model')
        ae_vis.visualize_best_samples(AE_Model.name, 'no2id_', 'NO Model')

        # get the best Epoch and Visualize the best Images
        with open(ae_eval.training_history_dir + '/Hist_NO', "rb") as fp:
            history_id = pickle.load(fp)

        min_id_value = min(history_id['val_loss'])
        min_id_epoch = history_id['val_loss'].index(min_id_value)
        print('Best Validation Epoch: ', min_id_epoch, ' with value: ', min_id_value)

        ae_eval.evaluate_model_on_testdata_chunk_gen_only(AE_Model.Call().predict, AE_Model.name, xb_val, xa_val, -1, '', 'NO Model')
        ae_vis.visualize_chunk_data_test(AE_Model.name, xb_val, xa_val, -1, '', 'NO Model')

def finalize_GAN_train(gan_args):
    gan_config_name = gan_args[0]
    vae_config_name = gan_args[1]
    degraded_dataset = gan_args[2]

    gan_config = read_cfg(gan_config_name, '../Model/Configs')
    ae_config = read_cfg(vae_config_name, '../Model/Configs')
    if gan_config['MODEL_TYPE'] != 'GAN':
        print('This is a routine to train GANs. Config of non GAN model was passed.')
        print('Exiting ...')
        exit()

    Data_loader = OneD_Data_Loader(gan_config['DATASET'])
    error_mode = gan_config['METRIC']
    gan_eval = GAN_evaluation('../Evaluation Results/', gan_config['MODEL_NAME'], '../Model/Configs' + '/' + gan_config_name, Data_loader.config, error_mode)
    ae_eval = VAE_evaluation('../Evaluation Results/',  ae_config['MODEL_NAME'],  '../Model/Configs' + '/' + vae_config_name, Data_loader.config, error_mode)

    gan_vis = GAN_visualisation('../Evaluation Results/', gan_config['MODEL_NAME'], Data_loader.config, error_mode)

    # load Data ---------------------------------------------------------
    validation_split = gan_config['EVAL_SPLIT']
    xA_train, xB_train, xa_val, xb_val, xA_test, xB_test = Data_loader.Load_Data_Tensors(degraded_dataset, validation_split)
    xa_val = xa_val[0:int(xa_val.shape[0] * 0.05)]
    xb_val = xb_val[0:int(xa_val.shape[0] * 0.05)]


    # /load Data --------------------------------------------------------

    # --- Build-Model ---
    epochs_id_gan = gan_config['FULL_Training']['EPOCHS_ID']
    epochs_norm_gan = gan_config['FULL_Training']['EPOCHS_NO']
    b_size_id = gan_config['FULL_Training']['BATCH_SIZE_ID']
    b_size_no = gan_config['FULL_Training']['BATCH_SIZE_NO']


    GAN = Basic_AE_Gan(gan_config['MODEL_NAME'], xA_train[0].shape, gan_config, ae_config, gan_eval, ae_eval)


    if epochs_id_gan > 0:
        GAN.load('ID')

        # Evaluate on test set
        eval_score_no_id = GAN.AE_Model.Call().evaluate(xA_test, xA_test, b_size_id)
        #eval_score_no_no = GAN.AE_Model.Call().evaluate(xB_test, xA_test, b_size_no)
        print('Evaluation Score on Test-set, GAN ID-Model, ID->ID: ', eval_score_no_id)
        #print('Evaluation Score on Test-set, GAN ID-Model, NO->ID: ', eval_score_no_no)
        print('------------------------------------------------------------------')

        # sample Model
        gan_eval.sample_best_model_output(xA_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'id2id_', 'GAN ID Model')
        #gan_eval.sample_best_model_output(xB_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'no2id_', 'GAN ID Model')
        # visualize samples
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'id2id_', 'GAN ID Model')
        #gan_vis.visualize_best_samples(GAN.AE_Model.name, 'no2id_', 'GAN ID Model')

        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xa_val, xa_val, -1, '', 'GAN ID Model ID')
        #gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xb_val, xa_val, -1, '', 'GAN ID Model NO')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xa_val, xa_val, -1, '', 'GAN ID Model ID')
        #gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xb_val, xa_val, -1, '', 'GAN ID Model NO')



        GAN.load('ID_spec')
        # Evaluate on test set
        eval_score_no_id = GAN.AE_Model.Call().evaluate(xA_test, xA_test, b_size_id)
        #eval_score_no_no = GAN.AE_Model.Call().evaluate(xB_test, xA_test, b_size_no)
        print('Evaluation Score on Test-set, GAN ID-Model_spec, ID->ID: ', eval_score_no_id)
        #print('Evaluation Score on Test-set, GAN ID-Model_spec, NO->ID: ', eval_score_no_no)
        print('------------------------------------------------------------------')

        # sample Model
        gan_eval.sample_best_model_output(xA_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'id2id_', 'GAN ID_spec Model')
        #gan_eval.sample_best_model_output(xB_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'no2id_', 'GAN ID_spec Model')
        # visualize samples
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'id2id_', 'GAN ID_spec Model')
        #gan_vis.visualize_best_samples(GAN.AE_Model.name, 'no2id_', 'GAN ID_spec Model')

        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xa_val, xa_val, -1, '', 'GAN ID_spec Model ID')
        #gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xb_val, xa_val, -1, '', 'GAN ID_spec Model NO')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xa_val, xa_val, -1, '', 'GAN ID_spec Model ID')
        #gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xb_val, xa_val, -1, '', 'GAN ID_spec Model NO')

    if epochs_norm_gan > 0:
        GAN.load('NO')

        # Evaluate on test set
        eval_score_no_id = GAN.AE_Model.Call().evaluate(xA_test, xA_test, b_size_no)
        eval_score_no_no = GAN.AE_Model.Call().evaluate(xB_test, xA_test, b_size_no)
        print('Evaluation Score on Test-set, GAN NO-Model, ID->ID: ', eval_score_no_id)
        print('Evaluation Score on Test-set, GAN NO-Model, NO->ID: ', eval_score_no_no)
        print('------------------------------------------------------------------')

        # sample Model
        gan_eval.sample_best_model_output(xA_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'id2id_', 'GAN NO Model')
        gan_eval.sample_best_model_output(xB_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'no2id_', 'GAN NO Model')
        # visualize samples
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'id2id_', 'GAN NO Model')
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'no2id_', 'GAN NO Model')

        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xa_val, xa_val, -1, '', 'GAN NO Model ID')
        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xb_val, xa_val, -1, '', 'GAN NO Model NO')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xa_val, xa_val, -1, '', 'GAN NO Model ID')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xb_val, xa_val, -1, '', 'GAN NO Model NO')



        GAN.load('NO_spec')
        # Evaluate on test set
        eval_score_no_id = GAN.AE_Model.Call().evaluate(xA_test, xA_test, b_size_no)
        eval_score_no_no = GAN.AE_Model.Call().evaluate(xB_test, xA_test, b_size_no)
        print('Evaluation Score on Test-set, GAN NO-Model_spec, ID->ID: ', eval_score_no_id)
        print('Evaluation Score on Test-set, GAN NO-Model_spec, NO->ID: ', eval_score_no_no)
        print('------------------------------------------------------------------')

        # sample Model
        gan_eval.sample_best_model_output(xA_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'id2id_', 'GAN NO_spec Model')
        gan_eval.sample_best_model_output(xB_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'no2id_', 'GAN NO_spec Model')
        # visualize samples
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'id2id_', 'GAN NO_spec Model')
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'no2id_', 'GAN NO_spec Model')

        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xa_val, xa_val, -1, '', 'GAN NO_spec Model ID')
        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xb_val, xa_val, -1, '', 'GAN NO_spec Model NO')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xa_val, xa_val, -1, '', 'GAN NO_spec Model ID')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xb_val, xa_val, -1, '', 'GAN NO_spec Model NO')

    if epochs_id_gan > 0:
        # evaluate Last Model
        GAN.load('LAST')

        # Evaluate on test set
        eval_score_no_id = GAN.AE_Model.Call().evaluate(xA_test, xA_test, b_size_id)
        print('Evaluation Score on Test-set, NO-Model, ID->ID: ', eval_score_no_id)
        print('------------------------------------------------------------------')

        # sample Model
        gan_eval.sample_best_model_output(xA_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'id2id_', 'LAST Model')
        # visualize samples
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'id2id_', 'LAST Model')

        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xa_val, xa_val, -1, '', 'LAST Model ID')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xa_val, xa_val, -1, '', 'LAST Model ID')

    if epochs_norm_gan > 0:
        # evaluate Last Model
        GAN.load('LAST')

        # Evaluate on test set
        eval_score_no_id = GAN.AE_Model.Call().evaluate(xA_test, xA_test, b_size_id)
        eval_score_no_no = GAN.AE_Model.Call().evaluate(xB_test, xA_test, b_size_no)
        print('Evaluation Score on Test-set, NO-Model, ID->ID: ', eval_score_no_id)
        print('Evaluation Score on Test-set, NO-Model, NO->ID: ', eval_score_no_no)
        print('------------------------------------------------------------------')

        # sample Model
        gan_eval.sample_best_model_output(xA_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'id2id_', 'LAST Model')
        gan_eval.sample_best_model_output(xB_test, xA_test, GAN.AE_Model.Call().predict, GAN.AE_Model.name, 'no2id_', 'LAST Model')
        # visualize samples
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'id2id_', 'LAST Model')
        gan_vis.visualize_best_samples(GAN.AE_Model.name, 'no2id_', 'LAST Model')

        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xa_val, xa_val, -1, '', 'LAST Model ID')
        gan_eval.evaluate_gan_on_testdata_chunk(GAN.AE_Model.Call().predict, GAN.AE_Model.name, GAN.discriminator.predict, [GAN.patch_length_width, GAN.patch_width], xb_val, xa_val, -1, '', 'LAST Model NO')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xa_val, xa_val, -1, '', 'LAST Model ID')
        gan_vis.visualize_chunk_data_test(GAN.AE_Model.name, xb_val, xa_val, -1, '', 'LAST Model NO')



    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    with open(gan_eval.chunk_eval_dir + '/misc_data', "rb") as fp:
        to_Save_in_misc = pickle.load(fp)

    best_gan_epoch_id_data, best_gan_epoch_no_data, best_gan_epoch_id_data_spec, best_gan_epoch_no_data_spec, start_time, train_time_end = to_Save_in_misc
    if epochs_id_gan > 0:
        gan_eval.best_epoch_id_GAN, gan_eval.best_loss_id_GAN = best_gan_epoch_id_data
        gan_eval.best_epoch_id_GAN_close2point5, gan_eval.gan_valid_loss_id_close2point5 = best_gan_epoch_id_data_spec

        print('Best ID Epoch: ', gan_eval.best_epoch_id_GAN, '  with ', gan_eval.best_loss_id_GAN)
        print('(special_metric: ', gan_eval.best_epoch_id_GAN_close2point5, '  with ', gan_eval.gan_valid_loss_id_close2point5, ')')
        print()

    if epochs_norm_gan > 0:
        gan_eval.best_epoch_no_GAN, gan_eval.best_loss_no_GAN = best_gan_epoch_no_data
        gan_eval.best_epoch_no_GAN_close2point5, gan_eval.gan_valid_loss_no_close2point5 = best_gan_epoch_no_data_spec

        print('Best NO Epoch: ', gan_eval.best_epoch_no_GAN, '  with ', gan_eval.best_loss_no_GAN)
        print('(special_metric: ', gan_eval.best_epoch_no_GAN_close2point5, '  with ', gan_eval.gan_valid_loss_no_close2point5, ')')
        print('-------------------------------------------------------------------------------------------------------------------')

    print('Training Times: ')
    print('Start: ', start_time, '  End: ', train_time_end)
    print('Duration: ', (train_time_end-start_time).total_seconds() / 3600, ' hours')


if __name__=='__main__':

    #args = sys.argv[1:]#['Basic_nRES_AE_Sins_ID_04.ini', 'B1']#['GAN_tst_cft.ini', 'AE_tst_cft.ini', 'B']#sys.argv[1:]

    args = ['GAN_Sqrs_ID_01.ini', 'GAN_nRES_AE_Sqrs_ID_01_pretrain.ini', 'B1']#['GAN_nRes_AE_Sqrs_ID_01.ini','Basic_nRES_AE_Sqrs_ID_01.ini', 'B1']
    if len(args) > 2:
        # its a GAN
        finalize_GAN_train(args)
    else:
        # its a VAE
        #finalize_VAE_train_onlyD(args)
        finalize_VAE_train(args)

    keras.backend.clear_session()




