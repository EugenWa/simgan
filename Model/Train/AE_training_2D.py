# import default path
import sys
sys.path.insert(0, '../../')

from Datamanager.Data_loader import Data_Loader
from Evaluation.Train_eval_2D import VAE_evaluation
from Model.Modules.AutoEncoders.Basic_2D_AutoEncoders import Basic_nRES_AE_2D
from Model.Modules.AutoEncoders.Basic_2D_Residual_AutoEncoders import Basic_nRES_ResidualAE_2D
from Utils.cfg_utils import read_cfg

# - keras -
import keras
import datetime
import pickle

if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = 0

    # generator
    config_name         = sys.argv[1] #'Basic_nRES_AE_Sins_ID_01.ini'#'AE_tst_cft.ini'#sys.argv[1]       # 'vae_cfg_1D.ini'
    degraded_dataset    = sys.argv[2]

    ae_config = read_cfg(config_name, '../Configs')
    if ae_config['MODEL_TYPE'] != 'Vae':
        print('This is a routine to train (Variational) Auto Encoders. Config of non (V)-AE model was passed.')
        print('Exiting ...')
        exit()
    print('Dataset to be operated on: ', ae_config['DATASET'])
    Data_load   = Data_Loader(ae_config['DATASET'])
    ae_eval     = VAE_evaluation('../../Evaluation Results/', ae_config['MODEL_NAME'],'../Configs' + '/' + config_name)


    # load Data ---------------------------------------------------------
    validation_split = ae_config['EVAL_SPLIT']
    classify_mode    = ae_config['FEATURE_CLASSIFY']

    xA_train_img_only, xA_target, xB_train, xa_val_img_only, xa_val_target, xb_val, xA_test_img_only, xA_test_target, xB_test = Data_load.Load_Data_Tensor_Split(degraded_dataset, validation_split, 0.1, False)

    # Load Params
    ae_name                     = ae_config['MODEL_NAME']
    model_identification_num    = ae_config['VAE']['MODEL_ID']
    epochs_id                   = ae_config['Training']['EPOCHS_ID']
    epochs_norm                 = ae_config['Training']['EPOCHS_NO']
    b_size_id                   = ae_config['Training']['BATCH_SIZE_ID']
    b_size_no                   = ae_config['Training']['BATCH_SIZE_NO']
    LOAD_PRE_T_ID               = ae_config['Training']['LOAD_PRE_T_ID']
    LOAD_PRE_T_NO               = ae_config['Training']['LOAD_PRE_T_NO']

    # --- Models ---
    Models = [Basic_nRES_AE_2D, Basic_nRES_ResidualAE_2D]
    AE_Model = Models[model_identification_num](ae_name, xA_train_img_only[0].shape, ae_config, ae_eval)

    start_time = datetime.datetime.now()    # time logging
    # train gen, ID
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if LOAD_PRE_T_ID:
        print('Loading Pretrained ID-Model:')
        AE_Model.load_pretrained_model('ID')
    if epochs_id > 0:
        print('Training ID ....')
        AE_Model.fit_ID(xA_target, xA_train_img_only, xa_val_target, xa_val_img_only, xA_test_target, xA_test_img_only, epochs_id, b_size_id, True, 1, True)

    # train gen, NO
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if LOAD_PRE_T_NO:
        print('Loading Pretrained NO-Model:')
        AE_Model.load_pretrained_model('NO')
    if epochs_norm > 0:
        print('Training NO ....')
        AE_Model.fit_NO(xA_target, xB_train, xa_val_target, xb_val, xA_test_target, xB_test, epochs_norm, b_size_no, True, 1, True)

    end_time = datetime.datetime.now()  # time logging

    print('Train start time: ', start_time)
    print('Train end   time: ', end_time)
    duration_time = (end_time-start_time).total_seconds()
    print('Duration: ', int(duration_time/3600), ' hours and ', (duration_time/60)%60, ' minutes')

    to_Save_in_misc = [start_time, end_time]
    with open(ae_eval.chunk_eval_dir + '/misc_data', "wb") as fp:
        pickle.dump(to_Save_in_misc, fp)
    # clean
    del AE_Model
    keras.backend.clear_session()








