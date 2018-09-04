from Data_loader import Data_Loader
from Train_eval import Generator_evaluation
from Generators1 import build_generator_conserving_ImgDim_basic, build_generator_conserving_ImgDim_one_ResBlock, build_generator_conserving_ImgDim_two_ResBlock
from Generators2 import build_generator_upscale_basic, build_generator_upscale_mtpl_resblocks
from Generators3 import build_generator_first, build_refiner_type_generator, build_generator_full_res, build_generator_full_res_nobias_first

# - keras -
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Input
import keras
CUDA_VISIBLE_DEVICES=0
from VAE_Models import Basic_VAE
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

    generator_name  = sys.argv[1]
    model_id        = int(sys.argv[2])
    lr              = float(sys.argv[3])
    lr_decay        = float(sys.argv[4])
    epochs_id       = int(sys.argv[5])
    epochs_norm     = int(sys.argv[6])

    # generator
    gen_eval = Generator_evaluation(generator_name)
    ####################
    vae = Basic_VAE(generator_name, xA_train[0].shape, lr_options=[keras.optimizers.adam, lr, lr_decay])
    ####################

    # set equal shuffling
    np.random.seed(12345)
    equal_shuffle = []
    for i in range(epochs_id + epochs_norm):            # shuffle all here in case the model uses random operations
        a = np.arange(xA_train.shape[0])                # which another doesnt -> rnd order caused by the seed will be permuted
        np.random.shuffle(a)
        equal_shuffle.append(a)

    # set training parameters
    train_shuffle = True
    batch_size = 28
    number_of_iterations = int(xA_train.shape[0]/batch_size)

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
    with open(gen_eval.test_path +'/' + vae.name + '_Mean_loss', "wb") as fp:
        pickle.dump([id_loss_epoch, ges_loss_epoch], fp)

    with open(gen_eval.test_path +'/' + vae.name + '_VAL_loss', "wb") as fp:
        pickle.dump([valid_loss_id, valid_loss_no], fp)


    # --- Sample Best Model ---
    vae.load_Model(gen_eval.test_path)

    #
    test_loss_id = gen_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_ID_img_only, vae.VAE_ID.name, xA_test, xA_test, 100001, 'ID')
    test_loss_no = gen_eval.evaluate_model_on_testdata_chunk_gen_only(vae.predict_NO_img_only, vae.VAE_NO.name, xB_test, xA_test, 100001)
    print('Test loss ID: ', test_loss_id)
    print('Test loss NO: ', test_loss_no)

    gen_eval.sample_best_model_output(xA_test, xA_test, vae.predict_ID_img_only, vae.VAE_ID.name, 'ID' )
    gen_eval.sample_best_model_output(xB_test, xA_test, vae.predict_NO_img_only, vae.VAE_NO.name)















