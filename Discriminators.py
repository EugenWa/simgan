# - keras -
import keras
from keras.layers import BatchNormalization, Conv2D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling2D, \
    UpSampling2D, Concatenate, Dropout, Flatten, Dense
from keras.models import Model, load_model
import numpy as np
CUDA_VISIBLE_DEVICES=0



def discriminator_build_4conv(disc_input, use_batch_normalisation=True, use_dropout=False):
    if use_batch_normalisation:
        disc_input = BatchNormalization()(disc_input)

    # convolve the input
    c_1 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(disc_input)
    c_1 = LeakyReLU(0.2)(c_1)
    c_1 = MaxPooling2D()(c_1)

    c_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_1)
    c_2 = LeakyReLU(0.2)(c_2)
    c_2 = MaxPooling2D()(c_2)
    if use_dropout:
        c_2 = Dropout(0.2)(c_2)
    if use_batch_normalisation:
        c_2 = BatchNormalization()(c_2)


    c_3 = Conv2D(15, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_2)
    c_3 = LeakyReLU(0.2)(c_3)
    c_3 = MaxPooling2D()(c_3)
    if use_dropout:
        c_3 = Dropout(0.2)(c_3)
    if use_batch_normalisation:
        c_3 = BatchNormalization()(c_3)

    c_4 = Conv2D(21, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_3)
    c_4 = LeakyReLU(0.2)(c_4)
    c_4 = MaxPooling2D()(c_4)
    if use_dropout:
        c_4 = Dropout(0.2)(c_4)
    if use_batch_normalisation:
        c_4 = BatchNormalization()(c_4)

    feature_vec = Flatten()(c_4)
    dn_1 = Dense(20)(feature_vec)
    dn_1 = LeakyReLU(0.2)(dn_1)

    disc_out = Dense(1, activation='sigmoid')(dn_1)
    return disc_out

from Data_loader import Data_Loader
if __name__=='__main__':
    inp = Input(shape=(64, 64, 1))
    d = discriminator_build_4conv(inp, True, True)
    m = Model(inp, d, name='tst')
    m.summary()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    D_loader = Data_Loader('triangles_64_pertL')
    xA_train, xB_train = D_loader.Load_Data_Tensors('train', invert=True)
    xA_test, xB_test = D_loader.Load_Data_Tensors('test', invert=True)

    all_data = np.zeros((xA_train.shape[0]+xB_train.shape[0], xA_train.shape[1], xA_train.shape[2], xA_train.shape[3]))
    all_data[0:xA_train.shape[0]] = xA_train
    all_data[xA_train.shape[0]:] = xB_train
    labels = np.zeros((xA_train.shape[0]+xB_train.shape[0],))
    labels[0:xA_train.shape[0]] = np.ones((xA_train.shape[0], ))

    shffle = np.arange(xA_train.shape[0]+xB_train.shape[0])
    np.random.shuffle(shffle)
    all_data = all_data[shffle]
    labels = labels[shffle]

    m.fit(all_data, labels, 128, 50)

    all_dataT = np.zeros(
        (xA_test.shape[0] + xB_test.shape[0], xA_test.shape[1], xA_test.shape[2], xA_test.shape[3]))
    all_dataT[0:xA_test.shape[0]] = xA_test
    all_dataT[xA_test.shape[0]:] = xB_test
    labels_T = np.zeros((xA_test.shape[0] + xB_test.shape[0],))
    labels_T[0:xA_test.shape[0]] = np.ones((xA_test.shape[0],))
    test_score = m.evaluate(all_dataT, labels_T)

    print('Eval score: ',test_score)
    '''
        32/4310 [..............................] - ETA: 14s
     800/4310 [====>.........................] - ETA: 0s 
    1568/4310 [=========>....................] - ETA: 0s
    2304/4310 [===============>..............] - ETA: 0s
    3040/4310 [====================>.........] - ETA: 0s
    3776/4310 [=========================>....] - ETA: 0s
    4310/4310 [==============================] - 0s 96us/step
    Eval score:  [0.008706285449846855, 0.9962877030162413]
    '''


