from Data_loader import Data_Loader
from Train_eval import GAN_evaluation
from Generators1 import build_generator_conserving_ImgDim_basic, build_generator_conserving_ImgDim_one_ResBlock, build_generator_conserving_ImgDim_two_ResBlock
from Generators2 import build_generator_upscale_basic, build_generator_upscale_mtpl_resblocks
from Generators3 import build_generator_first, build_refiner_type_generator, build_generator_full_res, build_generator_full_res_nobias_first

# - keras -
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
from keras.layers import Input, Lambda, Average
import keras
CUDA_VISIBLE_DEVICES=0
from VAE_Models import Basic_VAE
from Discriminators import discriminator_build_4conv
import numpy as np
import pickle

# - misc -
import sys


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES=0

    # load Data
    D_loader = Data_Loader('triangles_64_pertL')
    xA_train, xB_train = D_loader.Load_Data_Tensors('train', invert=True)
    xA_test, xB_test = D_loader.Load_Data_Tensors('test', invert=True)

    validation_split = 0.85

    xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
    xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

    xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
    xB_test = xB_test[int(xB_test.shape[0] * validation_split):]

    generator_name  ='tsttyststs'#= sys.argv[1]
    model_id        =0#= int(sys.argv[2])
    lr              =0.002#= float(sys.argv[3])
    lr_decay        =0.5#= float(sys.argv[4])
    epochs_id       =1#= int(sys.argv[5])
    epochs_norm     =1#= int(sys.argv[6])
    epochs_id_gan   =3#= int(sys.argv[7])
    epochs_norm_gan =6#= int(sys.argv[8])

    # generator
    gen_eval = GAN_evaluation(generator_name)
    ####################
    vae = Basic_VAE(generator_name, xA_train[0].shape, lr_options=[keras.optimizers.adam, lr, lr_decay])
    ####################
    # create patch_gan
    patch_lenght_width = (2 ** 1)
    patch_lenght_hight = (2 ** 1)
    patch_width = int(xA_train[0].shape[0]/patch_lenght_width)
    patch_hight = int(xA_train[0].shape[1]/patch_lenght_hight)

    disc_input = Input(shape=(patch_width, patch_hight, xA_train[0].shape[2]))
    # --- build_model ---
    disc_output = discriminator_build_4conv(disc_input, True, True)
    disc_optimizer = keras.optimizers.adam(lr, lr_decay)
    discriminator = Model(disc_input, disc_output, name='discriminator')
    discriminator.compile(optimizer=disc_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
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
        for p_j in range(patch_lenght_hight):
            disc_id_patch_eval.append(discriminator(Lambda(lambda x : x[:, p_i * patch_width:(p_i + 1) * patch_width, p_j * patch_hight:(p_j + 1) * patch_hight, :])(id_trafo)))
            disc_no_patch_eval.append(discriminator(Lambda(lambda x : x[:, p_i * patch_width:(p_i + 1) * patch_width, p_j * patch_hight:(p_j + 1) * patch_hight, :])(no_trafo)))
    if len(disc_id_patch_eval) is 1:
        disc_eval_id = disc_id_patch_eval[0]
        disc_eval_no = disc_no_patch_eval[0]
    else:
        disc_eval_id = Average()(disc_id_patch_eval)
        disc_eval_no = Average()(disc_no_patch_eval)
    vae_gan_id = Model(inputs=[full_model_ID_inp], outputs=[id_trafo, disc_eval_id])
    vae_gan_no = Model(inputs=[full_model_NO_inp], outputs=[no_trafo, features, disc_eval_no])

    # compile
    vae_gan_id_opt = keras.optimizers.adam(lr, lr_decay)
    vae_gan_no_opt = keras.optimizers.adam(lr, lr_decay)
    vae_gan_id.compile(optimizer=vae_gan_id_opt, loss=['mae', 'mae'], loss_weights=[10, 1])
    vae_gan_no.compile(optimizer=vae_gan_no_opt, loss=['mae', 'mse', 'mae'], loss_weights=[10, 7, 1])

    # set training parameters
    train_shuffle = True
    batch_size = 28
    number_of_iterations = int(xA_train.shape[0] / batch_size)

    #############################
    # --- Generator History --- #
    #############################
    gen_hist_len = 4
    Generator_history = np.zeros((batch_size * number_of_iterations * gen_hist_len, xA_train.shape[1], xA_train.shape[2] , xA_train.shape[3]))

    # set equal shuffling
    np.random.seed(12345)
    equal_shuffle = []
    for i in range(epochs_id + epochs_norm):            # shuffle all here in case the model uses random operations
        a = np.arange(xA_train.shape[0])                # which another doesnt -> rnd order caused by the seed will be permuted
        np.random.shuffle(a)
        equal_shuffle.append(a)

    # # #
    best_loss_id = np.inf
    best_loss_no = np.inf
    best_epoch_id = 0
    best_epoch_no = 0
    # # #
    # loss history
    id_loss_epoch = []
    ges_loss_epoch = []
    valid_loss_id = []
    valid_loss_no = []
    for epoch in range(epochs_id + epochs_norm):
        # shuffle set
        if train_shuffle:
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

        id_loss_batch = []
        ges_loss_batch = []
        for batch_i in range(number_of_iterations):
            xA_batch = xA_train[batch_i*batch_size:(batch_i+1)*batch_size]
            xB_batch = xB_train[batch_i*batch_size:(batch_i+1)*batch_size]
            loss_ges, loss_no = 0, 0
            if epoch < epochs_id:
                loss_id, _, _ = vae.train_model_on_batch(xA_batch, xA_batch, True)
                print('Epoch ', epoch, '/', (epochs_id + epochs_norm), '; Batch ', batch_i,'/', number_of_iterations, ' ID-Loss: ', loss_id)
            else:
                loss_ges, loss_id, loss_no =vae.train_model_on_batch(xB_batch, xA_batch, False)
                print('Epoch ', epoch, '/', (epochs_id + epochs_norm), '; Batch ', batch_i, '/', number_of_iterations, ':')
                print('Ges-Loss:  ', loss_ges)
                print('ID-Loss:   ', loss_id)
                print('Map-Loss:  ', loss_no)

            id_loss_batch.append(loss_id)
            ges_loss_batch.append(loss_ges)

        # calculate training loss
        id_loss_epoch.append(np.mean(id_loss_batch))
        ges_loss_epoch.append((np.mean(ges_loss_batch)))

        # evaluate Model on Validation_set
        validation_loss_id_epoch = gen_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name, xa_val, xa_val, epoch, 'ID')
        valid_loss_id.append(validation_loss_id_epoch)
        if epoch >= epochs_id:
            validation_loss_no_epoch = gen_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name, xb_val, xa_val, epoch)
            valid_loss_no.append(validation_loss_no_epoch)

            if best_loss_no > validation_loss_no_epoch:
                best_loss_no = validation_loss_no_epoch
                best_epoch_no = epoch
                # safe model
                print('SAVING')
                vae.save_model(gen_eval.test_path)




    # save history
    with open(gen_eval.test_path +'/' + vae.name + '_GOMean_loss', "wb") as fp:
        pickle.dump([id_loss_epoch, ges_loss_epoch], fp)

    with open(gen_eval.test_path +'/' + vae.name + '_GOVAL_loss', "wb") as fp:
        pickle.dump([valid_loss_id, valid_loss_no], fp)



    # ------ train GAN on that INIT ------
    # load the best vae model so far and continue training on it
    vae.load_Model(gen_eval.test_path)
    # # #
    best_loss_id = np.inf
    best_loss_no = [np.inf, np.inf]
    best_loss_disc_topoint5 = np.inf
    best_epoch_id = 0
    best_epoch_no = 0
    # # #
    # loss history
    gan_id_loss_epoch = []
    gan_ges_loss_epoch = []
    disc_loss_epoch = []
    gan_valid_loss_id = []
    gan_valid_loss_no = []

    # set equal shuffling
    np.random.seed(123456789)
    equal_shuffle = []
    for i in range(epochs_id_gan + epochs_norm_gan):    # shuffle all here in case the model uses random operations
        a = np.arange(xA_train.shape[0])        # which another doesnt -> rnd order caused by the seed will be permuted
        np.random.shuffle(a)
        equal_shuffle.append(a)

    gen_history_pointer = 0
    for epoch in range(epochs_id_gan + epochs_norm_gan):
        # shuffle set
        if train_shuffle:
            xA_train = xA_train[equal_shuffle[epoch]]
            xB_train = xB_train[equal_shuffle[epoch]]

        gan_id_loss_batch = []
        gan_ges_loss_batch = []
        disc_loss_batch = []
        for batch_i in range(number_of_iterations):
            xA_batch = xA_train[batch_i*batch_size:(batch_i+1)*batch_size]
            xB_batch = xB_train[batch_i*batch_size:(batch_i+1)*batch_size]

            loss_ges, loss_no = 0, 0
            normal_ptr = 0
            if epoch < epochs_id:
                loss_id = vae_gan_id.train_on_batch([xA_batch], [xA_batch, np.ones((xA_batch.shape[0], ))])
                print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i,'/', number_of_iterations, ' ID-Loss: ', loss_id)
                gan_id_loss_batch.append(loss_id)
                imgs_idvae = vae.VAE_ID.predict(xA_batch)
                imgs_novae = np.zeros((0, xA_batch.shape[1], xA_batch.shape[2], xA_batch.shape[3]))
                # fill of id images happens after if else cause id images are always generated
            else:
                # keep training id mapping
                loss_id = vae_gan_id.train_on_batch([xA_batch], [xA_batch, np.ones((xA_batch.shape[0],))])
                print('Epoch ', epoch, '/', (epochs_id_gan + epochs_norm_gan), '; Batch ', batch_i, '/',
                      number_of_iterations, ' ID-Loss: ', loss_id)
                gan_id_loss_batch.append(loss_id)
                # compute features
                imgs_idvae, features_id = vae.VAE_ID_MultiOut.predict(xA_batch)
                loss_ges =vae_gan_no.train_on_batch([xB_batch], [xA_batch, features_id ,np.ones((xA_batch.shape[0], ))])
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

            # FILL Gen history
            if gen_history_pointer * xA_batch.shape[0] < Generator_history.shape[0]:
                Generator_history[gen_history_pointer*xA_batch.shape[0]:(gen_history_pointer + 1)*xA_batch.shape[0]] = imgs_idvae
                gen_history_pointer += 1
                #id_imgs_from_gen_history = np.zeros((0, xA_batch.shape[1], xA_batch.shape[2], xA_batch.shape[3]))
                # imgs_idvae = imgs_idvae
            else:
                id_tmp = Generator_history[normal_ptr*int(xA_batch.shape[0]/2): (normal_ptr+1) * int(xA_batch.shape[0]/2)]
                Generator_history[normal_ptr * int(xA_batch.shape[0]/2):(normal_ptr + 1) * int(xA_batch.shape[0]/2)] = imgs_idvae[0:int(xA_batch.shape[0]/2)]
                normal_ptr += 1
                imgs_idvae[0:int(xA_batch.shape[0]/2)] = id_tmp
                imgs_idvae[int(xA_batch.shape[0]/2):]  = imgs_idvae[int(xA_batch.shape[0]/2):]


            # train discriminator
            imgs_ideal = xA_batch
            disc_batch = np.zeros((xA_batch.shape[0] + imgs_idvae.shape[0] + imgs_novae.shape[0], xA_batch.shape[1], xA_batch.shape[2], xA_batch.shape[3]))
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
                for p_j in range(patch_lenght_hight):
                    disc_loss.append(discriminator.train_on_batch(disc_batch[:, p_i * patch_width:(p_i + 1) * patch_width, p_j * patch_hight:(p_j + 1) * patch_hight ,:], disc_labels))
            disc_loss_batch.append(np.mean(disc_loss, axis=0))

            # shuffle GEN history, important, cause fill attatches new gen images at the beginning
            np.random.shuffle(Generator_history)

        # calculate training loss
        gan_id_loss_epoch.append(np.mean(gan_id_loss_batch, axis=0))
        gan_ges_loss_epoch.append((np.mean(gan_ges_loss_batch, axis=0)))


        # evaluate Model on Validation_set
        validation_loss_id_epoch = gen_eval.evaluate_gan_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name, discriminator.predict, [patch_lenght_width, patch_lenght_hight, patch_width, patch_hight] , xa_val, xa_val, epoch, 'ID')
        valid_loss_id.append(validation_loss_id_epoch)
        if epoch >= epochs_id:
            validation_loss_no_epoch = gen_eval.evaluate_gan_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name, discriminator.predict, [patch_lenght_width, patch_lenght_hight, patch_width, patch_hight], xb_val, xa_val, epoch)
            valid_loss_no.append(validation_loss_no_epoch)

            if best_loss_no[0] > validation_loss_no_epoch[0] and (best_loss_no[1] > np.abs(validation_loss_no_epoch[1] - 0.5)):
                best_loss_no = validation_loss_no_epoch
                best_epoch_no = epoch
                # safe model
                print('SAVING')
                vae.save_model(gen_eval.test_path, 'EvalM')
                discriminator.save(gen_eval.test_path + '/discriminator_EvalM.h5')
                vae_gan_id.save(gen_eval.test_path + '/VAE_GAN_ID_EvalM.h5')
                vae_gan_no.save(gen_eval.test_path + '/VAE_GAN_NO_EvalM.h5')

    # safe last model, due to the fact that quantitative decisions about quality are often misleading
    vae.save_model(gen_eval.test_path, 'LAST')
    discriminator.save(gen_eval.test_path + '/discriminator_LAST.h5')
    vae_gan_id.save(gen_eval.test_path + '/VAE_GAN_ID_LAST.h5')
    vae_gan_no.save(gen_eval.test_path + '/VAE_GAN_NO_LAST.h5')

    #
    test_loss_id = gen_eval.evaluate_gan_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name,
                                                                    discriminator.predict,
                                                                    [patch_lenght_width, patch_lenght_hight,
                                                                     patch_width, patch_hight], xA_test, xA_test, 199999,
                                                                    'ID_LAST')
    test_loss_no = gen_eval.evaluate_gan_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name,
                                                                    discriminator.predict,
                                                                    [patch_lenght_width, patch_lenght_hight,
                                                                     patch_width, patch_hight], xB_test, xA_test, 199999,
                                                                    'LAST')
    print('Test loss ID: ', test_loss_id)
    print('Test loss NO: ', test_loss_no)

    gen_eval.sample_best_model_output(xA_test, xA_test, vae.predict_ID_img_only, vae.VAE_ID.name, 'ID_LAST')
    gen_eval.sample_best_model_output(xB_test, xA_test, vae.predict_NO_img_only, vae.VAE_NO.name, 'LAST')


    ##########################################
    print('eVALUATE VIA METRIC:')

    # --- Sample Best Model ---

    del discriminator
    # this might be unnecessary
    del vae_gan_id
    del vae_gan_no
    vae.load_Model(gen_eval.test_path)
    discriminator = load_model(gen_eval.test_path + '/discriminator_LAST.h5')
    #vae_gan_id = load_model(gen_eval.test_path + '/VAE_GAN_ID_LAST.h5')
    #vae_gan_no = load_model(gen_eval.test_path + '/VAE_GAN_NO_LAST.h5')


    #
    test_loss_id = gen_eval.evaluate_gan_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name, discriminator.predict, [patch_lenght_width, patch_lenght_hight, patch_width, patch_hight] , xA_test, xA_test, 100000, 'ID_MEVAL')
    test_loss_no = gen_eval.evaluate_gan_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name, discriminator.predict, [patch_lenght_width, patch_lenght_hight, patch_width, patch_hight], xB_test, xA_test, 100000, 'MEVAL')
    print('Test loss ID: ', test_loss_id)
    print('Test loss NO: ', test_loss_no)

    gen_eval.sample_best_model_output(xA_test, xA_test, vae.predict_ID_img_only, vae.VAE_ID.name, 'ID_MEVAL' )
    gen_eval.sample_best_model_output(xB_test, xA_test, vae.predict_NO_img_only, vae.VAE_NO.name, 'MEVAL')















