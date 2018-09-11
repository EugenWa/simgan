# import default path
import os
import sys
sys.path.insert(0, '../../')

from Datamanager.OneD_Data_Creator                      import OneD_Data_Loader
from Evaluation.Train_eval_1D_update                    import GAN_evaluation
from Model.Modules.VAEs.VAE_Models_1D                   import Basic_2Channel_VAE
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
    config_name      = sys.argv[1]  #'gan_cfg.ini'#sys.argv[1]
    degraded_dataset = sys.argv[2]
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
    xA_test = xA_train[0:int(xA_train.shape[0] * 0.1)]
    xB_test = xB_train[0:int(xB_train.shape[0] * 0.1)]

    xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
    xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

    xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
    xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
    # /load Data --------------------------------------------------------

    # Load Params
    gan_name        = gan_config['MODEL_NAME']
    vae_name        = gan_config['FULL_MODEL']['VAE_NAME']
    epochs_id_vae   = gan_config['Training']['EPOCHS_ID']
    epochs_norm_vae = gan_config['Training']['EPOCHS_NO']
    b_size_id_vae   = gan_config['Training']['BATCH_SIZE_ID']
    b_size_no_vae   = gan_config['Training']['BATCH_SIZE_NO']
    epochs_id_gan   = gan_config['FULL_Training']['EPOCHS_ID']
    epochs_norm_gan = gan_config['FULL_Training']['EPOCHS_NO']
    b_size_id       = gan_config['FULL_Training']['BATCH_SIZE_ID']
    b_size_no       = gan_config['FULL_Training']['BATCH_SIZE_NO']

    optimizers = {'adam': keras.optimizers.adam}

    ####################
    vae = Basic_2Channel_VAE(vae_name, xA_train[0].shape, gan_config)
    ####################
    # create patch_gan

    patch_lenght_width = gan_config['DISC']['PATCH_NUMBER_W']
    patch_width = int(xA_train[0].shape[0]/patch_lenght_width)

    disc_input = Input(shape=(patch_width, xA_train[0].shape[1]))
    # --- build_model ---
    disc_output = discriminator_build_4conv_oneD(disc_input, 0.3, True, True)
    lr          = gan_config['DISC']['LEARNING_RATE']
    lr_decay    = gan_config['DISC']['LR_DEF']
    disc_optimizer = optimizers[gan_config['DISC']['OPTIMIZER']](lr, lr_decay)
    discriminator = Model(disc_input, disc_output, name=gan_config['FULL_MODEL']['DISC_NAME'])
    discriminator.compile(optimizer=disc_optimizer, loss=gan_config['DISC']['LOSS'], metrics=['accuracy'])
    discriminator.trainable = False

    # Build-full-Model
    full_model_ID_inp = Input(shape=xA_train[0].shape)
    full_model_NO_inp = Input(shape=xA_train[0].shape)

    id_trafo = vae.VAE_ID(full_model_ID_inp)
    no_trafo, features = vae.VAE_NO(full_model_NO_inp)
    ############################################################
    # ADD Patchwise!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ############################################################
    disc_id_patch_eval = []
    disc_no_patch_eval = []
    for p_i in range(patch_lenght_width):
            disc_id_patch_eval.append(discriminator(Lambda(lambda x : x[:, p_i * patch_width:(p_i + 1) * patch_width, :])(id_trafo)))
            disc_no_patch_eval.append(discriminator(Lambda(lambda x : x[:, p_i * patch_width:(p_i + 1) * patch_width, :])(no_trafo)))
    if len(disc_id_patch_eval) is 1:
        disc_eval_id = disc_id_patch_eval[0]
        disc_eval_no = disc_no_patch_eval[0]
    else:
        disc_eval_id = Average()(disc_id_patch_eval)
        disc_eval_no = Average()(disc_no_patch_eval)
    vae_gan_id = Model(inputs=[full_model_ID_inp], outputs=[id_trafo, disc_eval_id])
    vae_gan_no = Model(inputs=[full_model_NO_inp], outputs=[no_trafo, features, disc_eval_no])

    # compile

    vae_gan_id_opt = optimizers[gan_config['VAE']['OPTIMIZER']](0.1, 0.5)#(gan_config['VAE']['LEARNING_RATE'], gan_config['VAE']['LR_DEF'])
    vae_gan_no_opt = optimizers[gan_config['VAE']['OPTIMIZER']](0.1, 0.5)#(gan_config['VAE']['LEARNING_RATE'], gan_config['VAE']['LR_DEF'])
    vae_gan_id.compile(optimizer=vae_gan_id_opt, loss=[gan_config['FULL_MODEL']['IMG_LOSS'], gan_config['FULL_MODEL']['DISC_LOSS']], loss_weights=gan_config['FULL_MODEL']['LOSS_WEIGHTS_ID'])
    vae_gan_no.compile(optimizer=vae_gan_no_opt, loss=[gan_config['FULL_MODEL']['IMG_LOSS'], gan_config['FULL_MODEL']['FEATURE_LOSS'], gan_config['FULL_MODEL']['DISC_LOSS']], loss_weights=gan_config['FULL_MODEL']['LOSS_WEIGHTS_NO'])

    # set training parameters
    train_shuffle = True

    #############################
    # --- Generator History --- #
    #############################
    batch_size = max(b_size_id, b_size_no)
    Generator_history = np.zeros((batch_size * (int(xA_train.shape[0] / batch_size)) * gan_config['FULL_Training']['GENHIST_SIZE'], xA_train.shape[1], xA_train.shape[2]))


    # set equal shuffling
    np.random.seed(gan_config['SEED'])
    equal_shuffle = []
    for i in range(epochs_id_vae + epochs_norm_vae):            # shuffle all here in case the model uses random operations
        a = np.arange(xA_train.shape[0])                # which another doesnt -> rnd order caused by the seed will be permuted
        np.random.shuffle(a)
        equal_shuffle.append(a)

    for epoch in range(epochs_id_vae + epochs_norm_vae):
        if epoch < epochs_id_vae:
            batch_size = b_size_id_vae
        else:
            batch_size = b_size_no_vae
        number_of_iterations = 3#int(xA_train.shape[0] / batch_size)

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

            if epoch < epochs_id_vae:
                loss_id, _, _ = vae.train_model_on_batch(xA_batch, xA_batch, True)
                print('Epoch ', epoch, '/', (epochs_id_vae + epochs_norm_vae), '; Batch ', batch_i,'/', number_of_iterations, ' ID-Loss: ', loss_id)
            else:
                loss_ges, loss_id, loss_no =vae.train_model_on_batch(xB_batch, xA_batch, False)
                print('Epoch ', epoch, '/', (epochs_id_vae + epochs_norm_vae), '; Batch ', batch_i, '/', number_of_iterations, ':')
                print('Ges-Loss:  ', loss_ges)
                print('ID-Loss:   ', loss_id)
                print('Map-Loss:  ', loss_no)
                no_loss_batch.append(loss_no)
                ges_loss_batch.append(loss_ges)

            id_loss_batch.append(loss_id)

        # calculate training loss
        if epoch >= epochs_id_vae:
            gan_eval.no_loss_epoch.append([np.mean(no_loss_batch), np.std(no_loss_batch)])
            gan_eval.ges_loss_epoch.append([np.mean(ges_loss_batch), np.std(ges_loss_batch)])
        gan_eval.id_loss_epoch.append([np.mean(id_loss_batch), np.std(id_loss_batch)])


        # evaluate Model on Validation_set
        validation_loss_id_epoch = gan_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name, xa_val, xa_val, epoch, 'ID')#(vae.predict_ID_img_only, vae.VAE_ID.name, xa_val, xa_val, epoch, 'ID')
        gan_eval.valid_loss_id.append(validation_loss_id_epoch)
        # save best id model
        if gan_eval.q_vae_save_model_ID(validation_loss_id_epoch, epoch):
            # safe model
            print('SAVING ID-Model')
            vae.save_model(gan_eval.model_saves_dir, obj='VAE_ID')

        if epoch >= epochs_id_vae:
            validation_loss_no_epoch = gan_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name, xb_val, xa_val, epoch, 'NO') # vae.predict_NO_img_only
            gan_eval.valid_loss_no.append(validation_loss_no_epoch)

            if gan_eval.q_vae_save_model_NO(validation_loss_no_epoch, epoch):
                # safe model
                print('SAVING')
                vae.save_model(gan_eval.model_saves_dir, obj='VAE_NO')

    # save training history
    gan_eval.dump_training_history_ID()



    # ------ train GAN on that INIT ------
    # load the best vae model so far and continue training on it
    continue_training_no = True

    if continue_training_no:
        vae.load_Model(gan_eval.model_saves_dir, obj='VAE_NO')
    else:
        vae.load_Model(gan_eval.model_saves_dir, obj='VAE_ID')



    # set equal shuffling
    np.random.seed(gan_config['SEED'] + gan_config['SEED'])
    equal_shuffle = []
    for i in range(epochs_id_gan + epochs_norm_gan):    # shuffle all here in case the model uses random operations
        a = np.arange(xA_train.shape[0])        # which another doesnt -> rnd order caused by the seed will be permuted
        np.random.shuffle(a)
        equal_shuffle.append(a)

    gen_history_pointer = 0
    for epoch in range(epochs_id_gan + epochs_norm_gan):
        if epoch < epochs_id_gan:
            batch_size = b_size_id
        else:
            batch_size = b_size_no
        number_of_iterations = 3#int(xA_train.shape[0] / batch_size)

        # shuffle set
        if train_shuffle:
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

        gan_id_loss_batch = []
        gan_ges_loss_batch = []
        gan_no_loss_batch = []
        disc_loss_batch = []
        for batch_i in range(number_of_iterations):
            xA_batch = xA_train[batch_i*batch_size:(batch_i+1)*batch_size]
            xB_batch = xB_train[batch_i*batch_size:(batch_i+1)*batch_size]

            normal_ptr = 0
            if epoch < epochs_id_gan:
                loss_id = vae_gan_id.train_on_batch([xA_batch], [xA_batch, np.ones((xA_batch.shape[0], ))])
                print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i,'/', number_of_iterations, ' ID-Loss: ', loss_id)
                gan_id_loss_batch.append(loss_id)
                imgs_idvae = vae.VAE_ID.predict(xA_batch)
                imgs_novae = np.zeros((0, xA_batch.shape[1], xA_batch.shape[2]))
                # fill of id images happens after if else cause id images are always generated
            else:
                # keep training id mapping
                loss_id = vae_gan_id.train_on_batch([xA_batch], [xA_batch, np.ones((xA_batch.shape[0],))])
                print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i, '/',
                      number_of_iterations, ' ID-Loss: ', loss_id)
                gan_id_loss_batch.append(loss_id)
                # compute features
                imgs_idvae, features_id = vae.VAE_ID_MultiOut.predict(xA_batch)
                loss_ges = vae_gan_no.train_on_batch([xB_batch], [xA_batch, features_id ,np.ones((xA_batch.shape[0], ))])
                print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i, '/', number_of_iterations, ':')
                print('Ges-Loss:  ', loss_ges)
                gan_ges_loss_batch.append(loss_ges)
                imgs_novae, _ = vae.VAE_NO.predict(xB_batch)
                if gen_history_pointer*xA_batch.shape[0] < Generator_history.shape[0]:
                    # just fill
                    Generator_history[gen_history_pointer * xA_batch.shape[0]:(gen_history_pointer + 1) * xA_batch.shape[0]] = imgs_novae
                    gen_history_pointer += 1
                    #no_imgs_from_gen_history = np.zeros((0, xA_batch.shape[1], xA_batch.shape[2], xA_batch.shape[3]))
                    #imgs_novae = imgs_novae
                else:
                    no_tmp = Generator_history[0:int(xA_batch.shape[0]/2)]
                    Generator_history[normal_ptr * int(xA_batch.shape[0]/2):(normal_ptr + 1) * int(xA_batch.shape[0]/2)] = imgs_novae[0:int(xA_batch.shape[0]/2)]
                    normal_ptr += 1
                    imgs_novae[0:int(xA_batch.shape[0]/2)] = no_tmp
                    imgs_novae[int(xA_batch.shape[0]/2):]  = imgs_novae[int(xA_batch.shape[0]/2):]

            # --- FILL Gen history ---
            if gen_history_pointer * xA_batch.shape[0] < Generator_history.shape[0]:
                Generator_history[gen_history_pointer*xA_batch.shape[0]:(gen_history_pointer + 1)*xA_batch.shape[0]] = imgs_idvae #id_imgs_from_gen_history = np.zeros((0, xA_batch.shape[1], xA_batch.shape[2], xA_batch.shape[3]))
                gen_history_pointer += 1                                                                                          # imgs_idvae = imgs_idvae

            else:
                id_tmp = Generator_history[normal_ptr*int(xA_batch.shape[0]/2): (normal_ptr+1) * int(xA_batch.shape[0]/2)]
                Generator_history[normal_ptr * int(xA_batch.shape[0]/2):(normal_ptr + 1) * int(xA_batch.shape[0]/2)] = imgs_idvae[0:int(xA_batch.shape[0]/2)]
                normal_ptr += 1
                imgs_idvae[0:int(xA_batch.shape[0]/2)] = id_tmp
                imgs_idvae[int(xA_batch.shape[0]/2):]  = imgs_idvae[int(xA_batch.shape[0]/2):]


            # --- train discriminator ---
            imgs_ideal = xA_batch
            disc_batch = np.zeros((xA_batch.shape[0] + imgs_idvae.shape[0] + imgs_novae.shape[0], xA_batch.shape[1], xA_batch.shape[2]))
            disc_batch[0:imgs_ideal.shape[0]] = imgs_ideal
            disc_batch[imgs_ideal.shape[0]:imgs_ideal.shape[0]+imgs_idvae.shape[0]] = imgs_idvae
            disc_batch[imgs_ideal.shape[0]+imgs_idvae.shape[0]:] = imgs_novae
            disc_labels = np.zeros((xA_batch.shape[0] + imgs_idvae.shape[0] + imgs_novae.shape[0],))
            disc_labels[0:imgs_ideal.shape[0]] = np.ones((imgs_ideal.shape[0],))

            shuffle_batch = np.arange(disc_labels.shape[0])
            np.random.shuffle(shuffle_batch)
            disc_labels=disc_labels[shuffle_batch]
            disc_batch = disc_batch[shuffle_batch]

            disc_loss = []
            for p_i in range(patch_lenght_width):
                    disc_loss.append(discriminator.train_on_batch(disc_batch[:, p_i * patch_width:(p_i + 1) * patch_width,:], disc_labels))
            disc_loss_batch.append(np.mean(disc_loss, axis=0))


            # shuffle GEN history, important, cause fill attatches new gen images at the beginning
            np.random.shuffle(Generator_history)

        # calculate training loss

        gan_eval.gan_disc_loss_epoch.append([np.mean(disc_loss_batch, axis=0), np.std(disc_loss_batch, axis=0)])
        gan_eval.gan_id_loss_epoch.append([np.mean(gan_id_loss_batch, axis=0), np.std(gan_id_loss_batch, axis=0)])


        if epoch > epochs_id_gan:
            gan_eval.gan_ges_loss_epoch.append([np.mean(gan_ges_loss_batch, axis=0), np.std(gan_ges_loss_batch, axis=0)])


        # evaluate Model on Validation_set
        validation_loss_id_epoch = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict_ID_img_only, vae.VAE_ID.name, discriminator.predict, [patch_lenght_width,  patch_width] , xa_val, xa_val, epoch, 'ID')
        gan_eval.gan_valid_loss_id.append(validation_loss_id_epoch)
        print('--------------------------------------------------')
        print(validation_loss_id_epoch)
        print('--------------------------------------------------')
        if epoch >= epochs_id_gan:
            validation_loss_no_epoch = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict_NO_img_only, vae.VAE_NO.name, discriminator.predict, [patch_lenght_width,  patch_width], xb_val, xa_val, epoch)
            gan_eval.gan_valid_loss_no.append(validation_loss_no_epoch)

            if gan_eval.best_loss_no_GAN[0] > validation_loss_no_epoch[0] and (gan_eval.best_loss_no_GAN[1] > np.abs(validation_loss_no_epoch[1] - 0.5)):
                gan_eval.best_loss_no_GAN = validation_loss_no_epoch
                gan_eval.best_epoch_no_GAN = epoch

                # safe model
                print('SAVING')
                vae.save_model(gan_eval.model_saves_dir, 'EvalM')
                discriminator.save(gan_eval.model_saves_dir + '/discriminator_EvalM.h5')
                vae_gan_id.save(gan_eval.model_saves_dir+ '/VAE_GAN_ID_EvalM.h5')
                vae_gan_no.save(gan_eval.model_saves_dir + '/VAE_GAN_NO_EvalM.h5')

    # save history
    gan_eval.dump_training_history_GAN()

    # safe last model, due to the fact that quantitative decisions about quality are often misleading
    vae.save_model(gan_eval.model_saves_dir, 'LAST')
    discriminator.save(gan_eval.model_saves_dir + '/discriminator_LAST.h5')
    vae_gan_id.save(gan_eval.model_saves_dir + '/VAE_GAN_ID_LAST.h5')
    vae_gan_no.save(gan_eval.model_saves_dir + '/VAE_GAN_NO_LAST.h5')



    #
    print(' ---------------- TESTING ---------------- ')
    print()
    print('Test the LAST GAN-Model on ID_mapping:')
    test_loss_id = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict_ID_img_only, vae.VAE_ID.name,
                                                                    discriminator.predict,
                                                                    [patch_lenght_width,
                                                                     patch_width], xA_test, xA_test, 199999,
                                                                    'ID_LAST')
    test_loss_no = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict_NO_img_only, vae.VAE_NO.name,
                                                                    discriminator.predict,
                                                                    [patch_lenght_width,
                                                                     patch_width], xB_test, xA_test, 199999,
                                                                    'LAST')
    print('Test loss ID: ', test_loss_id)
    print('Test loss NO: ', test_loss_no)
    print(' ---------------- --- ---------------- ')

    gan_eval.sample_best_model_output(xA_test, xA_test, vae.predict_ID_img_only, vae.VAE_ID.name, 'ID_LAST')
    gan_eval.sample_best_model_output(xB_test, xA_test, vae.predict_NO_img_only, vae.VAE_NO.name, 'LAST')
    gan_eval.visualize_best_samples(vae.VAE_ID.name, 'ID_LAST')
    gan_eval.visualize_best_samples(vae.VAE_NO.name, 'LAST')



    ##########################################
    print('Test via EVALUATION METRIC:')

    # --- Sample Best Model ---

    del discriminator
    # this might be unnecessary
    del vae_gan_id
    del vae_gan_no
    vae.load_Model(gan_eval.model_saves_dir, 'EvalM')
    discriminator = load_model(gan_eval.model_saves_dir + '/discriminator_LAST.h5')
    #vae_gan_id = load_model(gen_eval.test_path + '/VAE_GAN_ID_LAST.h5')
    #vae_gan_no = load_model(gen_eval.test_path + '/VAE_GAN_NO_LAST.h5')


    #
    test_loss_id = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict_ID_img_only, vae.VAE_ID.name, discriminator.predict, [patch_lenght_width, patch_width] , xA_test, xA_test, 100000, 'ID_MEVAL')
    test_loss_no = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict_NO_img_only, vae.VAE_NO.name, discriminator.predict, [patch_lenght_width, patch_width], xB_test, xA_test, 100000, 'MEVAL')
    print('Test loss ID: ', test_loss_id)
    print('Test loss NO: ', test_loss_no)

    gan_eval.sample_best_model_output(xA_test, xA_test, vae.predict_ID_img_only, vae.VAE_ID.name, 'ID_MEVAL' )
    gan_eval.sample_best_model_output(xB_test, xA_test, vae.predict_NO_img_only, vae.VAE_NO.name, 'MEVAL')
    gan_eval.visualize_best_samples(vae.VAE_ID.name, 'ID_MEVAL' )
    gan_eval.visualize_best_samples(vae.VAE_NO.name, 'MEVAL')
    print(' ---------------- --- ---------------- ')

    del discriminator
    keras.backend.clear_session()















