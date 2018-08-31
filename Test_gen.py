from keras.models import load_model
import keras
CUDA_VISIBLE_DEVICES=0
from Data_loader import Data_Loader
from Train_eval import Generator_evaluation
import numpy as np
# - misc -
import sys




if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES=0

    # load Data
    D_loader = Data_Loader('triangles_64_pertL')
    xA_train, xB_train = D_loader.Load_Data_Tensors('train', invert=True)
    xA_test_t, xB_test_t = D_loader.Load_Data_Tensors('test', invert=True)

    validation_split = 0.85

    xa_val = xA_test_t[0:int(xA_test_t.shape[0] * validation_split)]
    xb_val = xB_test_t[0:int(xB_test_t.shape[0] * validation_split)]

    xA_test = xA_test_t[int(xA_test_t.shape[0] * validation_split):]
    xB_test = xB_test_t[int(xB_test_t.shape[0] * validation_split):]

    # generator
    generator_name = sys.argv[1]
    gen_eval = Generator_evaluation(generator_name)
    model_id = int(sys.argv[2])

    generator_A2B = load_model(generator_name + 'weights.h5')
    eval_score_norm = generator_A2B.evaluate(xA_test, xA_test, batch_size=128)
    print('ID-Evaluation: ',  eval_score_norm)
    eval_score_norm = generator_A2B.evaluate(xB_test, xA_test, batch_size=128)
    print('Normal-Evaluation: ', eval_score_norm)

    # make samples
    gen_eval.evaluate_model_on_testdata_chunk_gen_only(generator_A2B, xB_test, xA_test, 100001, True)
    gen_eval.sample_best_model_output(xB_test, xA_test, generator_A2B)

    del generator_A2B
    keras.backend.clear_session()
