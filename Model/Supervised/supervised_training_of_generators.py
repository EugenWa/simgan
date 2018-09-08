# import default path
import os
import sys
sys.path.insert(0, '../../')

from Datamanager.Data_loader    import Data_Loader
from Evaluation.Train_eval      import Generator_evaluation
from Model.Modules.Generators.Generators1   import build_generator_conserving_ImgDim_basic, build_generator_conserving_ImgDim_one_ResBlock, build_generator_conserving_ImgDim_two_ResBlock
from Model.Modules.Generators.Generators2   import build_generator_upscale_basic, build_generator_upscale_mtpl_resblocks
from Model.Modules.Generators.Generators3   import build_generator_first, build_refiner_type_generator, build_generator_full_res, build_generator_full_res_nobias_first
from Utils.cfg_utils import read_cfg

# - keras -
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Input
import keras




if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES=0

    # generator
    config_name = sys.argv[1]
    generator_c = read_cfg(config_name, '../Configs')
    if generator_c['MODEL_TYPE'] != 'Gen':
        print('This is a routine to train generators. Config of non generator model was passed.')
        print('Exiting ...')
        exit()
    gen_eval = Generator_evaluation(generator_c['MODEL_NAME'])

    # load Data ---------------------------------------------------------
    D_loader = Data_Loader(generator_c['DATASET'])
    xA_train, xB_train = D_loader.Load_Data_Tensors('train', invert=True)
    xA_test, xB_test = D_loader.Load_Data_Tensors('test', invert=True)

    validation_split = generator_c['EVAL_SPLIT']

    xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
    xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

    xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
    xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
    # /load Data --------------------------------------------------------

    ####################
    generators = [build_generator_conserving_ImgDim_basic, build_generator_conserving_ImgDim_one_ResBlock,
                  build_generator_conserving_ImgDim_two_ResBlock, build_generator_upscale_basic,
                  build_generator_upscale_mtpl_resblocks, build_generator_first, build_refiner_type_generator,
                  build_generator_full_res, build_generator_full_res_nobias_first]
    optimizers = {'adam':keras.optimizers.adam}
    ####################

    # Generator Parameters
    generator_name  = generator_c['MODEL_NAME']
    model_id        = generator_c['Generator']['MODEL_ID']
    lr              = generator_c['Generator']['LEARNING_RATE']
    lr_decay        = generator_c['Generator']['LR_DEF']
    epochs_id       = generator_c['Training']['EPOCHS_ID']
    epochs_norm     = generator_c['Training']['EPOCHS_NO']
    b_size_id       = generator_c['Training']['BATCH_SIZE_ID']
    b_size_no       = generator_c['Training']['BATCH_SIZE_NO']
    # - Misc parameters
    model_save_dir = gen_eval.test_path + '/' + generator_c['MODEL_SAVE_DIR']
    os.makedirs(model_save_dir, exist_ok = True)

    # --- build gen ---
    gen_input = Input(shape=xA_train[0].shape)
    generator_A2B = generators[model_id](gen_input, generator_name)
    optimizer = optimizers[generator_c['Generator']['OPTIMIZER']](lr, lr_decay)

    generator_A2B.compile(optimizer=optimizer, loss=generator_c['Generator']['LOSS'], loss_weights=generator_c['Generator']['LOSS_WEIGHTS'])#, metrics=['mean_absolute_error', 'val_mean_absolute_error'])

    # train gen, ID
    checkpointer0 = ModelCheckpoint(model_save_dir + '/' + generator_name + '_weights.h5', verbose=1, save_best_only=True)
    generator_A2B.fit(xA_train, xA_train, validation_data=(xa_val, xa_val), shuffle=True, epochs=epochs_id, batch_size=b_size_id, callbacks=[checkpointer0])
    eval_score_id = generator_A2B.evaluate(xA_test, xA_test, batch_size=b_size_id)

    # train gen, no
    del generator_A2B
    generator_A2B = load_model(model_save_dir + '/' + generator_name + '_weights.h5')
    checkpointer = ModelCheckpoint(model_save_dir + '/' + generator_name + '_weights.h5', verbose=1, save_best_only=True)
    generator_A2B.fit(xB_train, xA_train, validation_data=(xb_val, xa_val), shuffle=True, epochs=epochs_norm, batch_size=b_size_no, callbacks=[checkpointer])
    eval_score_norm = generator_A2B.evaluate(xB_test, xA_test, batch_size=b_size_no)
    print(eval_score_norm)

    # evaluate the best model
    del generator_A2B
    generator_A2B = load_model(model_save_dir + '/' + generator_name + '_weights.h5')
    eval_score_norm = generator_A2B.evaluate(xB_test, xA_test, batch_size=b_size_no)
    print(eval_score_norm)

    # make samples
    gen_eval.evaluate_model_on_testdata_chunk_gen_only(generator_A2B.predict, generator_A2B.name, xB_test, xA_test, -1, obj='Eval_on_test')
    gen_eval.sample_best_model_output(xB_test, xA_test, generator_A2B.predict, generator_A2B.name)

    del generator_A2B
    keras.backend.clear_session()






