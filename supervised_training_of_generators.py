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

    # generator
    generator_name = sys.argv[1]
    gen_eval = Generator_evaluation(generator_name)


    ####################
    generators = [build_generator_conserving_ImgDim_basic, build_generator_conserving_ImgDim_one_ResBlock,
                  build_generator_conserving_ImgDim_two_ResBlock, build_generator_upscale_basic,
                  build_generator_upscale_mtpl_resblocks, build_generator_first, build_refiner_type_generator,
                  build_generator_full_res, build_generator_full_res_nobias_first]
    ####################

    model_id = int(sys.argv[2])
    lr = float(sys.argv[3])
    lr_decay =  float(sys.argv[4])
    epochs_id = int(sys.argv[5])
    epochs_norm = int(sys.argv[6])
    # --- build gen ---
    gen_input = Input(shape=xA_train[0].shape)
    generator_A2B = generators[model_id](gen_input, generator_name)
    optimizer = keras.optimizers.adam(lr, lr_decay)

    generator_A2B.compile(optimizer=optimizer, loss=['mae'])#, metrics=['mean_absolute_error', 'val_mean_absolute_error'])

    # train gen, ID
    checkpointer0 = ModelCheckpoint(generator_name + 'weights.h5', verbose=1, save_best_only=True)
    generator_A2B.fit(xA_train, xA_train, validation_data=(xa_val, xa_val), shuffle=True, epochs=epochs_id, batch_size=28, callbacks=[checkpointer0])
    eval_score_id = generator_A2B.evaluate(xA_test, xA_test, batch_size=128)
    print(eval_score_id)

    del generator_A2B
    generator_A2B = load_model(generator_name + 'weights.h5')
    checkpointer = ModelCheckpoint(generator_name + 'weights.h5', verbose=1, save_best_only=True)
    generator_A2B.fit(xB_train, xA_train, validation_data=(xb_val, xa_val), shuffle=True, epochs=epochs_norm, batch_size=28, callbacks=[checkpointer])
    del generator_A2B
    generator_A2B = load_model(generator_name + 'weights.h5')
    eval_score_norm = generator_A2B.evaluate(xB_test, xA_test, batch_size=128)
    print(eval_score_norm)

    # make samples
    gen_eval.evaluate_model_on_testdata_chunk_gen_only(generator_A2B, xB_test, xA_test, 100001, True)
    gen_eval.sample_best_model_output(xB_test, xA_test, generator_A2B)

    del generator_A2B
    keras.backend.clear_session()






