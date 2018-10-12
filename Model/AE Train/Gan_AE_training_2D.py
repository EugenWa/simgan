# import default path
import sys
sys.path.insert(0, '../../')

from Datamanager.Data_loader                            import Data_Loader
from Evaluation.Train_eval_2D                           import GAN_evaluation, VAE_evaluation
from Model.Modules.GANs.Basic_AE_Gan import Basic_AE_Gan
from Utils.cfg_utils import read_cfg

# - keras -
import keras
import numpy as np
import random
import pickle
import datetime


def create_disc_batch(imgs_ideal, imgs_from_ae):
    c_disc_batch = np.zeros((imgs_ideal.shape[0] + imgs_from_ae.shape[0], imgs_ideal.shape[1], imgs_ideal.shape[2]))
    c_disc_batch[0:imgs_ideal.shape[0]] = imgs_ideal
    c_disc_batch[imgs_ideal.shape[0]:] = imgs_from_ae
    c_disc_labels = np.zeros((c_disc_batch.shape[0],))
    c_disc_labels[0:imgs_ideal.shape[0]] = np.ones((imgs_ideal.shape[0],))

    shuffle_batch = np.arange(c_disc_labels.shape[0])
    np.random.shuffle(shuffle_batch)
    c_disc_labels = c_disc_labels[shuffle_batch]
    c_disc_batch  = c_disc_batch[shuffle_batch]

    return c_disc_batch, c_disc_labels



if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # generator
    config_name_gan      = 'GAN_tst_cft.ini'#sys.argv[1]  #'gan_cfg.ini'#sys.argv[1] #'gan_cfg_IDo_1.ini'
    config_name_ae       = 'AE_tst_cft.ini'#sys.argv[2]
    degraded_dataset     = 'B'#sys.argv[3]

    gan_config = read_cfg(config_name_gan, '../Configs')
    ae_config  = read_cfg(config_name_ae,  '../Configs')
    if gan_config['MODEL_TYPE'] != 'GAN':
        print('This is a routine to train GANs. Config of non GAN model was passed.')
        print('Exiting ...')
        exit()

    Data_loader = Data_Loader(gan_config['DATASET'])
    error_mode = gan_config['METRIC']
    gan_eval    = GAN_evaluation('../../Evaluation Results/', gan_config['MODEL_NAME'], '../Configs' + '/' + config_name_gan, error_mode)
    ae_eval     = VAE_evaluation('../../Evaluation Results/', ae_config['MODEL_NAME'],  '../Configs' + '/' + config_name_ae,  error_mode)


    # load Data ---------------------------------------------------------
    validation_split = gan_config['EVAL_SPLIT']
    xA_train_img_only, xA_target, xB_train, xa_val_img_only, xa_val_target, xb_val, xA_test_img_only, xA_test_target, xB_test = Data_loader.Load_Data_Tensor_Split(degraded_dataset, validation_split, 0)
    xA_train = xA_target
    xa_val = xa_val_target
    xA_test = xA_test_target
    # /load Data --------------------------------------------------------

    # --- Build-Model ---
    GAN = Basic_AE_Gan(gan_config['MODEL_NAME'], xA_train[0].shape, gan_config, ae_config, gan_eval, ae_eval)


    # Load Params
    epochs_id_ae    = ae_config['Training']['EPOCHS_ID']
    epochs_norm_ae  = ae_config['Training']['EPOCHS_NO']
    b_size_id_ae    = ae_config['Training']['BATCH_SIZE_ID']
    b_size_no_ae    = ae_config['Training']['BATCH_SIZE_NO']

    epochs_id_gan   = gan_config['FULL_Training']['EPOCHS_ID']
    epochs_norm_gan = gan_config['FULL_Training']['EPOCHS_NO']
    b_size_id       = gan_config['FULL_Training']['BATCH_SIZE_ID']
    b_size_no       = gan_config['FULL_Training']['BATCH_SIZE_NO']


    # discriminator balancing
    disc_train_activation   = gan_config['FULL_Training']['DISC_TRAIN_ACTIVATION']
    disc_train_mode         = gan_config['FULL_Training']['DISC_TRAIN_MODE']
    disc_train_ApE          = gan_config['FULL_Training']['DISC_TRAIN_ApE']
    disc_sample_population  = range(GAN.patch_length_width)

    unsupervised_setting    = gan_config['FULL_MODEL']['UNSUPERVISED']




    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --- Training ---
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set training parameters
    train_shuffle = True
    #############################
    # --- Generator History --- #
    #############################
    batch_size = max(b_size_id, b_size_no)
    Generator_history = np.zeros((batch_size * (int(xA_train.shape[0] / batch_size)) * gan_config['FULL_Training']['GENHIST_SIZE'], xA_train.shape[1], xA_train.shape[2]))


    # -------------------------------------------------
    # remark models can be pretrained in AE_training.py
    # -------------------------------------------------
    if epochs_id_ae > 0:
        GAN.AE_Model.load_pretrained_model('ID')        # loads model from ae_eval Directory
        print('loaded Pretrained ID-Model')
    if epochs_norm_ae > 0:
        GAN.AE_Model.load_pretrained_model('NO')
        print('loaded Pretrained NO-Model')


    # check whether Model should be Pre-Loaded



    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --- GAN-Training ---
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # set equal shuffling
    np.random.seed(gan_config['SEED'])
    GAN_DISC_RATIO = gan_config['FULL_Training']['DISC_TRAIN_RATIO']
    GAN_validation_size = 0.03

    equal_shuffle = []
    for i in range(epochs_id_gan + epochs_norm_gan):    # shuffle all here in case the model uses random operations
        a = np.arange(xA_train.shape[0])                # which another doesnt -> rnd order caused by the seed will be permuted
        np.random.shuffle(a)
        equal_shuffle.append(a)

    np.random.seed(gan_config['SEED'])
    equal_shuffle_val_set = []
    for i in range(epochs_id_gan + epochs_norm_gan):
        a = np.arange(xa_val.shape[0])
        np.random.shuffle(a)
        equal_shuffle_val_set.append(a)


    gen_history_pointer = 0
    for epoch in range(epochs_id_gan + epochs_norm_gan):
        if epoch < epochs_id_gan:
            batch_size = b_size_id
        else:
            batch_size = b_size_no
        number_of_iterations = 7#int(xA_train.shape[0] / batch_size)

        # shuffle set
        if train_shuffle:
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

        gan_id_loss_batch = []
        gan_no_loss_batch = []
        disc_loss_batch   = []

        for batch_i in range(number_of_iterations):

            xA_batch = xA_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            xB_batch = xB_train[batch_i * batch_size:(batch_i + 1) * batch_size]

            if epoch < epochs_id_gan:
                if unsupervised_setting:
                    loss_id = GAN.GAN.train_on_batch([xA_batch], [np.ones((xA_batch.shape[0],))])
                else:
                    loss_id = GAN.GAN.train_on_batch([xA_batch], [xA_batch, np.ones((xA_batch.shape[0], ))])
                print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i, '/', number_of_iterations, ' ID-Loss: ', loss_id)
                gan_id_loss_batch.append(loss_id)
                imgs_vae = GAN.AE_Model.Call().predict(xA_batch)
            else:
                # keep training id mapping
                if unsupervised_setting:
                    loss_no = GAN.GAN.train_on_batch([xB_batch], [np.ones((xA_batch.shape[0],))])
                else:
                    loss_no = GAN.GAN.train_on_batch([xB_batch], [xA_batch, np.ones((xA_batch.shape[0],))])
                gan_no_loss_batch.append(loss_no)
                print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i, '/', number_of_iterations, ' NO-Loss: ', loss_no)
                imgs_vae = GAN.AE_Model.Call().predict(xB_batch)


            if gen_history_pointer*xA_batch.shape[0] < Generator_history.shape[0]:
                # just fill
                Generator_history[gen_history_pointer * xA_batch.shape[0]:(gen_history_pointer + 1) * xA_batch.shape[0]] = imgs_vae
                gen_history_pointer += 1
            else:
                no_tmp = Generator_history[0:int(xA_batch.shape[0]/2)]
                Generator_history[0:int(xA_batch.shape[0]/2)] = imgs_vae[0:int(xA_batch.shape[0]/2)]
                imgs_vae[0:int(xA_batch.shape[0]/2)] = no_tmp

            # shuffle GEN history, important, cause fill attatches new gen images at the beginning
            np.random.shuffle(Generator_history)


            if batch_i%GAN_DISC_RATIO == 0:
                # --- train discriminator ---
                disc_batch, disc_labels = create_disc_batch(xA_batch, imgs_vae)

                disc_loss = []
                if disc_train_mode == 'MIN':
                    p_i_S = random.sample(disc_sample_population, disc_train_ApE)
                    for p_i in p_i_S:
                        disc_loss.append(GAN.discriminator.train_on_batch(disc_batch[:, p_i * GAN.patch_width:(p_i + 1) * GAN.patch_width, :], disc_labels))
                else:
                    skip_disc_counter = 0
                    for p_i in range(GAN.patch_length_width):
                            if np.random.uniform(0, 1) < disc_train_activation:
                                skip_disc_counter += 1
                                continue
                            disc_loss.append(GAN.discriminator.train_on_batch(disc_batch[:, p_i * GAN.patch_width:(p_i + 1) * GAN.patch_width,:], disc_labels))
                    if skip_disc_counter >=  GAN.patch_length_width:
                        p_i = np.random.randint(0, GAN.patch_length_width)
                        disc_loss.append(GAN.discriminator.train_on_batch(disc_batch[:, p_i * GAN.patch_width:(p_i + 1) * GAN.patch_width, :], disc_labels))
                disc_loss_batch.append(np.mean(disc_loss, axis=0))



        # calculate training loss
        gan_eval.gan_disc_loss_epoch.append([np.mean(disc_loss_batch, axis=0), np.std(disc_loss_batch, axis=0)])

        if epoch < epochs_id_gan:
            gan_eval.gan_id_loss_epoch.append([np.mean(gan_id_loss_batch, axis=0), np.std(gan_id_loss_batch, axis=0)])
        else:
            gan_eval.gan_no_loss_epoch.append([np.mean(gan_no_loss_batch, axis=0), np.std(gan_no_loss_batch, axis=0)])



        # evaluate Model on Validation_set
        xa_val = xa_val[equal_shuffle_val_set[epoch]]
        xb_val = xb_val[equal_shuffle_val_set[epoch]]

        xa_val_red = xa_val[0:int(xa_val.shape[0] * GAN_validation_size)]
        xb_val_red = xb_val[0:int(xb_val.shape[0] * GAN_validation_size)]

        if epoch < epochs_id_gan:
            validation_loss_id_epoch = gan_eval.evaluate_gan_minimalistic(GAN.AE_Model.Call(), GAN.discriminator, xa_val_red, xa_val_red, [GAN.patch_length_width, GAN.patch_width])
            gan_eval.gan_valid_loss_id.append(validation_loss_id_epoch)
            print('--------------------------------------------------')
            print(validation_loss_id_epoch)
            print('--------------------------------------------------')
            if gan_eval.best_loss_id_GAN[0] > validation_loss_id_epoch[0]:
                gan_eval.best_loss_id_GAN = validation_loss_id_epoch
                gan_eval.best_epoch_id_GAN = epoch
                # save model
                print('Saving ID GAN')
                GAN.save('ID')

            if (gan_eval.gan_valid_loss_id_close2point5[0] > validation_loss_id_epoch[0]) and (np.abs(gan_eval.gan_valid_loss_id_close2point5[1][0]-0.5) > np.abs(validation_loss_id_epoch[1][0] - 0.5)):
                gan_eval.gan_valid_loss_id_close2point5 = validation_loss_id_epoch
                gan_eval.best_epoch_id_GAN_close2point5 = epoch
                # save model
                print('Saving ID GAN with Disc close to .5')
                GAN.save('ID_spec')


        else:
            validation_loss_no_epoch = gan_eval.evaluate_gan_minimalistic(GAN.AE_Model.Call(), GAN.discriminator, xb_val_red, xa_val_red, [GAN.patch_length_width, GAN.patch_width])
            gan_eval.gan_valid_loss_no.append(validation_loss_no_epoch)

            if gan_eval.best_loss_no_GAN[0] > validation_loss_no_epoch[0]:
                gan_eval.best_loss_no_GAN = validation_loss_no_epoch
                gan_eval.best_epoch_no_GAN = epoch

                # safe model
                print('SAVING NO GAN')
                GAN.save('NO')

            if gan_eval.gan_valid_loss_no_close2point5[0] > validation_loss_no_epoch[0] and (np.abs(gan_eval.gan_valid_loss_no_close2point5[1][0]-0.5) > np.abs(validation_loss_no_epoch[1][0] - 0.5)):
                gan_eval.gan_valid_loss_no_close2point5 = validation_loss_no_epoch
                gan_eval.best_epoch_no_GAN_close2point5 = epoch

                # safe model
                print('SAVING NO GAN with Disc close to .5')
                GAN.save('NO_spec')

    # save history
    gan_eval.dump_training_history_GAN()

    # safe last model, due to the fact that quantitative decisions about quality are often misleading
    GAN.save('LAST')

    train_time_end = datetime.datetime.now()

    best_gan_epoch_id_data = [gan_eval.best_epoch_id_GAN, gan_eval.best_loss_id_GAN]
    best_gan_epoch_no_data = [gan_eval.best_epoch_no_GAN, gan_eval.best_loss_no_GAN]

    best_gan_epoch_id_data_spec = [gan_eval.best_epoch_id_GAN_close2point5, gan_eval.gan_valid_loss_id_close2point5]
    best_gan_epoch_no_data_spec = [gan_eval.best_epoch_no_GAN_close2point5, gan_eval.gan_valid_loss_no_close2point5]

    to_Save_in_misc = [best_gan_epoch_id_data, best_gan_epoch_no_data, best_gan_epoch_id_data_spec, best_gan_epoch_no_data_spec, start_time, train_time_end]
    with open(gan_eval.chunk_eval_dir + '/misc_data', "wb") as fp:
        pickle.dump(to_Save_in_misc, fp)

    del GAN.discriminator
    del GAN.GAN
    keras.backend.clear_session()
