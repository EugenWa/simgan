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
import random
import pickle

# - misc -
import sys


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES = 0

    # generator
    config_name      = 'gan_cfg_oneD_1.ini'#sys.argv[1]  #'gan_cfg.ini'#sys.argv[1]
    degraded_dataset = 'B5'#sys.argv[2]
    gan_config = read_cfg(config_name, '../Configs')
    if gan_config['MODEL_TYPE'] != 'GAN':
        print('This is a routine to train GANs. Config of non GAN model was passed.')
        print('Exiting ...')
        exit()
    Data_loader = OneD_Data_Loader(gan_config['DATASET'])
    gan_eval = GAN_evaluation(gan_config['MODEL_NAME'], Data_loader.config, 1)

    # load Data ---------------------------------------------------------
    validation_split = gan_config['EVAL_SPLIT']

    xA_train = Data_loader.load_A()
    xB_train = Data_loader.load_B(degraded_dataset)


    xA_test = xA_train[0:int(xA_train.shape[0] * 0.1)]
    xB_test = xB_train[0:int(xB_train.shape[0] * 0.1)]

    xA_train = xA_train[int(xA_train.shape[0] * 0.1):]
    xB_train = xB_train[int(xB_train.shape[0] * 0.1):]

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
    id = ''
    lr_vae                  = gan_config['VAE' + id]['LEARNING_RATE']
    lr_decay_vae            = gan_config['VAE' + id]['LR_DEF']
    relu_param              = gan_config['VAE' + id]['RELU_PARAM']
    filters                 = gan_config['VAE' + id]['FILTERS']
    use_drop_out            = gan_config['VAE' + id]['USE_DROP_OUT']
    use_batch_normalisation = gan_config['VAE' + id]['USE_BATCH_NORM']
    trafo_layers            = gan_config['VAE' + id]['TRAFO_LAYERS']
    optimizer_vae           = optimizers[gan_config['VAE' + id]['OPTIMIZER']]

    # create patch_gan

    patch_lenght_width = gan_config['DISC']['PATCH_NUMBER_W']
    patch_width = int(xA_train[0].shape[0]/patch_lenght_width)

    # discriminator balancing
    disc_train_activation = gan_config['FULL_Training']['DISC_TRAIN_ACTIVATION']
    disc_train_mode = gan_config['FULL_Training']['DISC_TRAIN_MODE']
    disc_train_ApE = gan_config['FULL_Training']['DISC_TRAIN_ApE']
    disc_sample_population = range(patch_lenght_width)


    ### --- Construct Model ---
    disc_input = Input(shape=(patch_width, xA_train[0].shape[1]))
    # --- build_model ---
    disc_output = discriminator_build_4conv_oneD(disc_input, 0.3, True, True)
    lr          = gan_config['DISC']['LEARNING_RATE']
    lr_decay    = gan_config['DISC']['LR_DEF']
    disc_optimizer = optimizers[gan_config['DISC']['OPTIMIZER']](lr, lr_decay)
    discriminator = Model(disc_input, disc_output, name=gan_config['FULL_MODEL']['DISC_NAME'])
    discriminator.compile(optimizer=disc_optimizer, loss=gan_config['DISC']['LOSS'], metrics=['accuracy'])
    discriminator.trainable = False

    ####################
    vae = VAE_RES(vae_name, xA_train[0].shape, filters, trafo_layers, relu_param, use_batch_normalisation, use_drop_out)
    vae_optimizer = optimizer_vae(lr_vae, lr_decay_vae)
    vae.compile(optimizer=vae_optimizer, loss=gan_config['VAE' + id]['IMAGE_LOSS'])
    ####################

    gan_eval.visualize_chunk_data_test(vae.name, xb_val, xa_val, 22)
    exit()

    # Build-full-Model
    full_model_inp = Input(shape=xA_train[0].shape)

    id_trafo = vae(full_model_inp)
    ############################################################
    # ADD Patchwise!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ############################################################
    disc_patch_eval = []
    for p_i in range(patch_lenght_width):
            disc_patch_eval.append(discriminator(Lambda(lambda x : x[:, p_i * patch_width:(p_i + 1) * patch_width, :])(id_trafo)))
    if len(disc_patch_eval) is 1:
        disc_eval = disc_patch_eval[0]
    else:
        disc_eval = Average()(disc_patch_eval)

    vae_gan = Model(inputs=[full_model_inp], outputs=[id_trafo, disc_eval])

    # compile

    vae_gan_optimizer = optimizers[gan_config['VAE']['OPTIMIZER']](gan_config['VAE']['LEARNING_RATE'], gan_config['VAE']['LR_DEF'])
    vae_gan.compile(optimizer=vae_gan_optimizer, loss=[gan_config['FULL_MODEL']['IMG_LOSS'], gan_config['FULL_MODEL']['DISC_LOSS']], loss_weights=gan_config['FULL_MODEL']['LOSS_WEIGHTS_ID'])

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
        number_of_iterations = int(xA_train.shape[0] / batch_size)

        # shuffle set
        if train_shuffle:
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

        loss_batch_id = []
        loss_batch_no = []
        for batch_i in range(number_of_iterations):
            xA_batch = xA_train[batch_i*batch_size:(batch_i+1)*batch_size]
            xB_batch = xB_train[batch_i*batch_size:(batch_i+1)*batch_size]

            if epoch < epochs_id_vae:
                loss_id = vae.train_on_batch(xA_batch, xA_batch)
                print('Epoch ', epoch, '/', (epochs_id_vae + epochs_norm_vae), '; Batch ', batch_i,'/', number_of_iterations, ' ID-Loss: ', loss_id)
                loss_batch_id.append(loss_id)
            else:
                loss_no = vae.train_on_batch(xB_batch, xA_batch)
                print('Epoch ', epoch, '/', (epochs_id_vae + epochs_norm_vae), '; Batch ', batch_i, '/', number_of_iterations, ':')
                print('Map-Loss:  ', loss_no)
                loss_batch_no.append(loss_no)


        # calculate training loss
        if epoch < epochs_id_vae:
            gan_eval.id_loss_epoch.append([np.mean(loss_batch_id), np.std(loss_batch_id)])
        else:
            gan_eval.no_loss_epoch.append([np.mean(loss_batch_no), np.std(loss_batch_no)])


        # evaluate Model on Validation_set
        validation_loss_id_epoch = gan_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict, vae.name, xa_val, xa_val, epoch, 'ID')#(vae.predict_ID_img_only, vae.VAE_ID.name, xa_val, xa_val, epoch, 'ID')
        gan_eval.valid_loss_id.append(validation_loss_id_epoch)
        # save best id model
        if gan_eval.q_vae_save_model_ID(validation_loss_id_epoch, epoch):
            # safe model
            print('SAVING ID-Model')
            vae.save(gan_eval.model_saves_dir + '/VAE_ID')

        if epoch >= epochs_id_vae:
            validation_loss_no_epoch = gan_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict, vae.name, xb_val, xa_val, epoch, 'NO') # vae.predict_NO_img_only
            gan_eval.valid_loss_no.append(validation_loss_no_epoch)

            if gan_eval.q_vae_save_model_NO(validation_loss_no_epoch, epoch):
                # safe model
                print('SAVING')
                vae.save(gan_eval.model_saves_dir + '/VAE_NO')

    # save training history
    gan_eval.dump_training_history_ID(False)



    # ------ train GAN on that INIT ------
    # load the best vae model so far and continue training on it
    continue_training_no = True

    #if continue_training_no:
        #vae = load_model(gan_eval.model_saves_dir + '/VAE_NO')
    #else:
        #vae = load_model(gan_eval.model_saves_dir + '/VAE_ID')



    # set equal shuffling
    np.random.seed(gan_config['SEED'] + gan_config['SEED'])
    GAN_DISC_RATIO = gan_config['FULL_Training']['DISC_TRAIN_RATIO']
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
        number_of_iterations = int(xA_train.shape[0] / batch_size)

        # shuffle set
        if train_shuffle:
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

        gan_id_loss_batch = []
        gan_no_loss_batch = []
        disc_loss_batch = []
        for batch_i in range(number_of_iterations):

            xA_batch = xA_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            xB_batch = xB_train[batch_i * batch_size:(batch_i + 1) * batch_size]

            rnd_batches_range = range(number_of_iterations)
            randomly_selected_batches = random.sample(rnd_batches_range, GAN_DISC_RATIO)
            randomly_selected_batches[0] = batch_i
            for gen_ratio in range(GAN_DISC_RATIO):
                xA_batch_gen_only = xA_train[randomly_selected_batches[gen_ratio] * batch_size:(randomly_selected_batches[gen_ratio] + 1) * batch_size]
                xB_batch_gen_only = xB_train[randomly_selected_batches[gen_ratio] * batch_size:(randomly_selected_batches[gen_ratio] + 1) * batch_size]
                if epoch < epochs_id_gan:
                    loss_id = vae_gan.train_on_batch([xA_batch_gen_only], [xA_batch_gen_only, np.ones((xA_batch_gen_only.shape[0], ))])
                    print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i,'/', number_of_iterations, ' Batch_GEN: ', gen_ratio, ' ID-Loss: ', loss_id)
                    gan_id_loss_batch.append(loss_id)
                    imgs_vae = vae.predict(xA_batch)
                else:
                    # keep training id mapping
                    loss_no = vae_gan.train_on_batch([xB_batch_gen_only], [xA_batch_gen_only, np.ones((xA_batch_gen_only.shape[0],))])
                    gan_no_loss_batch.append(loss_no)
                    print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i, '/', number_of_iterations, ' Batch_GEN: ', gen_ratio, ' NO-Loss: ', loss_no)
                    imgs_vae = vae.predict(xB_batch_gen_only)

            if gen_history_pointer*xA_batch.shape[0] < Generator_history.shape[0]:
                # just fill
                Generator_history[gen_history_pointer * xA_batch.shape[0]:(gen_history_pointer + 1) * xA_batch.shape[0]] = imgs_vae
                gen_history_pointer += 1
                #no_imgs_from_gen_history = np.zeros((0, xA_batch.shape[1], xA_batch.shape[2], xA_batch.shape[3]))
                #imgs_novae = imgs_novae
            else:
                no_tmp = Generator_history[0:int(xA_batch.shape[0]/2)]
                Generator_history[0:int(xA_batch.shape[0]/2)] = imgs_vae[0:int(xA_batch.shape[0]/2)]
                imgs_vae[0:int(xA_batch.shape[0]/2)] = no_tmp
                imgs_vae[int(xA_batch.shape[0]/2):]  = imgs_vae[int(xA_batch.shape[0]/2):]



            # --- train discriminator ---
            imgs_ideal = xA_batch
            disc_batch = np.zeros((imgs_ideal.shape[0] + imgs_vae.shape[0], imgs_ideal.shape[1], imgs_ideal.shape[2]))
            disc_batch[0:imgs_ideal.shape[0]]   = imgs_ideal
            disc_batch[imgs_ideal.shape[0]:]    = imgs_vae
            disc_labels = np.zeros((imgs_ideal.shape[0] + imgs_vae.shape[0],))
            disc_labels[0:imgs_ideal.shape[0]] = np.ones((imgs_ideal.shape[0],))

            shuffle_batch   = np.arange(disc_labels.shape[0])
            np.random.shuffle(shuffle_batch)
            disc_labels     = disc_labels[shuffle_batch]
            disc_batch      = disc_batch[shuffle_batch]

            # ADD EVENTUALL A ROUTINE TO TRAIN DISC ON ONLY THIS PATCH
            disc_loss = []

            if disc_train_mode == 'MIN':
                p_i_S = random.sample(disc_sample_population, disc_train_ApE)
                for p_i in p_i_S:
                    disc_loss.append(discriminator.train_on_batch(disc_batch[:, p_i * patch_width:(p_i + 1) * patch_width, :], disc_labels))
            else:
                for p_i in range(patch_lenght_width):
                        if np.random.uniform(0, 1) < disc_train_activation:
                            continue
                        disc_loss.append(discriminator.train_on_batch(disc_batch[:, p_i * patch_width:(p_i + 1) * patch_width,:], disc_labels))
            disc_loss_batch.append(np.mean(disc_loss, axis=0))


            # shuffle GEN history, important, cause fill attatches new gen images at the beginning
            np.random.shuffle(Generator_history)

        # calculate training loss

        gan_eval.gan_disc_loss_epoch.append([np.mean(disc_loss_batch, axis=0), np.std(disc_loss_batch, axis=0)])

        if epoch < epochs_id_gan:
            gan_eval.gan_id_loss_epoch.append([np.mean(gan_id_loss_batch, axis=0), np.std(gan_id_loss_batch, axis=0)])
        else:
            gan_eval.gan_ges_loss_epoch.append([np.mean(gan_no_loss_batch, axis=0), np.std(gan_no_loss_batch, axis=0)])
            # saves in gesloss !!!

        # evaluate Model on Validation_set
        validation_loss_id_epoch = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict, vae.name, discriminator.predict, [patch_lenght_width,  patch_width] , xa_val, xa_val, epoch, 'ID')
        gan_eval.gan_valid_loss_id.append(validation_loss_id_epoch)
        print('--------------------------------------------------')
        print(validation_loss_id_epoch)
        print('--------------------------------------------------')
        if epoch >= epochs_id_gan:
            validation_loss_no_epoch = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict, vae.name, discriminator.predict, [patch_lenght_width,  patch_width], xb_val, xa_val, epoch)
            gan_eval.gan_valid_loss_no.append(validation_loss_no_epoch)

            if gan_eval.best_loss_no_GAN[0] > validation_loss_no_epoch[0] and (np.abs(gan_eval.best_loss_no_GAN[1]-0.5) > np.abs(validation_loss_no_epoch[1] - 0.5)):
                gan_eval.best_loss_no_GAN = validation_loss_no_epoch
                gan_eval.best_epoch_no_GAN = epoch

                # safe model
                print('SAVING')
                vae.save(gan_eval.model_saves_dir + '/EvalM')
                discriminator.save(gan_eval.model_saves_dir + '/discriminator_EvalM.h5')
                vae_gan.save(gan_eval.model_saves_dir + '/VAE_GAN_EvalM.h5')

    # save history
    # !!!!!!!"!"!?!"?!?"!??"?!"??!?"?!??"??"!?"?!?!
    gan_eval.dump_training_history_GAN()

    # safe last model, due to the fact that quantitative decisions about quality are often misleading
    vae.save(gan_eval.model_saves_dir + '/LAST')
    discriminator.save(gan_eval.model_saves_dir + '/discriminator_LAST.h5')
    vae_gan.save(gan_eval.model_saves_dir + '/VAE_GAN_LAST.h5')

    #
    print(' ---------------- TESTING ---------------- ')
    print()
    print('Test the LAST GAN-Model on ID_mapping:')
    test_loss_id = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict, vae.name,
                                                                    discriminator.predict,
                                                                    [patch_lenght_width,
                                                                     patch_width], xA_test, xA_test, -1,
                                                                    'ID_LAST')
    test_loss_no = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict, vae.name,
                                                                    discriminator.predict,
                                                                    [patch_lenght_width,
                                                                     patch_width], xB_test, xA_test, -1,
                                                                    'LAST')
    print('Test loss ID: ', test_loss_id)
    print('Test loss NO: ', test_loss_no)
    print(' ---------------- --- ---------------- ')

    gan_eval.sample_best_model_output(xA_test, xA_test, vae.predict, vae.name, 'ID_LAST')
    gan_eval.sample_best_model_output(xB_test, xA_test, vae.predict, vae.name, 'LAST')
    gan_eval.visualize_best_samples(vae.name, 'ID_LAST')
    gan_eval.visualize_best_samples(vae.name, 'LAST')



    ##########################################
    print('Test via EVALUATION METRIC:')

    # --- Sample Best Model ---

    del discriminator
    # this might be unnecessary
    del vae_gan
    vae = load_model(gan_eval.model_saves_dir + '/EvalM')
    discriminator = load_model(gan_eval.model_saves_dir + '/discriminator_LAST.h5')
    #vae_gan_id = load_model(gen_eval.test_path + '/VAE_GAN_ID_LAST.h5')
    #vae_gan_no = load_model(gen_eval.test_path + '/VAE_GAN_NO_LAST.h5')


    #
    test_loss_id = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict, vae.name, discriminator.predict, [patch_lenght_width, patch_width] , xA_test, xA_test, -1, 'ID_MEVAL')
    test_loss_no = gan_eval.evaluate_gan_on_testdata_chunk(vae.predict, vae.name, discriminator.predict, [patch_lenght_width, patch_width], xB_test, xA_test, -1, 'MEVAL')
    print('Test loss ID: ', test_loss_id)
    print('Test loss NO: ', test_loss_no)

    gan_eval.sample_best_model_output(xA_test, xA_test, vae.predict, vae.name, 'ID_MEVAL' )
    gan_eval.sample_best_model_output(xB_test, xA_test, vae.predict, vae.name, 'MEVAL')
    gan_eval.visualize_best_samples(vae.name, 'ID_MEVAL' )
    gan_eval.visualize_best_samples(vae.name, 'MEVAL')
    print(' ---------------- --- ---------------- ')
    print('best gan epoch: ', gan_eval.best_epoch_no_GAN)
    to_Save_in_misc = [gan_eval.best_epoch_no_GAN]
    with open(gan_eval.chunk_eval_dir + '/misc_data', "wb") as fp:
        pickle.dump(to_Save_in_misc, fp)

    del discriminator
    del vae
    keras.backend.clear_session()















