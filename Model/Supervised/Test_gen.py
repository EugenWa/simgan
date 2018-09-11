from keras.models import load_model
import keras
import os
import sys
sys.path.insert(0, '../../')

from Datamanager.Data_loader    import Data_Loader
from Evaluation.Train_eval_update      import Generator_evaluation
from Utils.cfg_utils import read_cfg




if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES=0

    # generator
    config_name = sys.argv[1]
    generator_c = read_cfg(config_name, '../Configs')
    if generator_c['MODEL_TYPE'] != 'Gen':
        print('This is a routine to test generators. Config of non generator model was passed.')
        print('Exiting ...')
        exit()
    gen_eval = Generator_evaluation(generator_c['MODEL_NAME'])
    generator_name  = generator_c['MODEL_NAME']
    b_size_id       = generator_c['Training']['BATCH_SIZE_ID']
    b_size_no       = generator_c['Training']['BATCH_SIZE_NO']

    # - Misc parameters
    model_save_dir = gen_eval.test_path + '/' + generator_c['MODEL_SAVE_DIR']
    os.makedirs(model_save_dir, exist_ok=True)


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


    # validate Model
    generator_A2B = load_model(model_save_dir + '/' + generator_name + '_weights.h5')
    eval_score_norm = generator_A2B.evaluate(xA_test, xA_test, batch_size=b_size_id)
    print('ID-Evaluation: ',  eval_score_norm)
    eval_score_norm = generator_A2B.evaluate(xB_test, xA_test, batch_size=b_size_no)
    print('Normal-Evaluation: ', eval_score_norm)

    # make samples
    gen_eval.evaluate_model_on_testdata_chunk_gen_only(generator_A2B.predict, generator_A2B.name, xB_test, xA_test, -1, obj='Test')
    gen_eval.sample_best_model_output(xB_test, xA_test, generator_A2B.predict, generator_A2B.name)

    del generator_A2B
    keras.backend.clear_session()
