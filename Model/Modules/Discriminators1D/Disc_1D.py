# - keras -
import keras
from keras.layers import BatchNormalization, Conv1D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling1D, \
    UpSampling2D, Concatenate, Dropout, Flatten, Dense
from keras.models import Model, load_model
import numpy as np

def discriminator_build_4conv_oneD(disc_input, relu_param=0.3, use_batch_normalisation=True, use_dropout=False):
    if use_batch_normalisation:
        disc_input = BatchNormalization()(disc_input)

    # convolve the input
    c_1 = Conv1D(4, kernel_size=5, strides=1, padding='same', use_bias=False)(disc_input)
    c_1 = LeakyReLU(relu_param)(c_1)
    c_1 = MaxPooling1D()(c_1)

    c_2 = Conv1D(8, kernel_size=3, strides=1, padding='same')(c_1)
    c_2 = LeakyReLU(relu_param)(c_2)
    c_2 = MaxPooling1D()(c_2)
    if use_dropout:
        c_2 = Dropout(0.2)(c_2)
    if use_batch_normalisation:
        c_2 = BatchNormalization()(c_2)


    c_3 = Conv1D(12, kernel_size=3, strides=1, padding='same')(c_2)
    c_3 = LeakyReLU(relu_param)(c_3)
    c_3 = MaxPooling1D()(c_3)
    if use_dropout:
        c_3 = Dropout(0.2)(c_3)
    if use_batch_normalisation:
        c_3 = BatchNormalization()(c_3)

    c_4 = Conv1D(16, kernel_size=3, strides=1, padding='same')(c_3)
    c_4 = LeakyReLU(relu_param)(c_4)
    c_4 = MaxPooling1D()(c_4)
    if use_dropout:
        c_4 = Dropout(0.2)(c_4)
    if use_batch_normalisation:
        c_4 = BatchNormalization()(c_4)

    c_4 = Conv1D(1, kernel_size=3, strides=1, padding='same')(c_4)
    c_4 = LeakyReLU(relu_param)(c_4)
    c_4 = MaxPooling1D()(c_4)
    feature_vec = Flatten()(c_4)
    dn_1 = Dense(20)(feature_vec)
    dn_1 = LeakyReLU(relu_param)(dn_1)

    disc_out = Dense(1, activation='sigmoid')(dn_1)
    return disc_out

def small_disc_1D(disc_input, relu_param=0.3, use_batch_normalisation=True, use_dropout=False):
    if use_batch_normalisation:
        disc_input = BatchNormalization()(disc_input)

    # convolve the input
    c_1 = Conv1D(18, kernel_size=3, strides=1, padding='same', use_bias=True)(disc_input)
    c_1 = LeakyReLU(relu_param)(c_1)
    c_1 = MaxPooling1D()(c_1)

    if use_batch_normalisation:
        c_2 = BatchNormalization()(c_1)


    feature_vec = Flatten()(c_2)
    dn_1 = Dense(20)(feature_vec)
    dn_1 = LeakyReLU(relu_param)(dn_1)

    dn_1 = Dense(20)(dn_1)
    dn_1 = LeakyReLU(relu_param)(dn_1)

    disc_out = Dense(1, activation='sigmoid')(dn_1)
    return disc_out

def disc_1D_full_dense(disc_input, relu_param=0.3, use_batch_normalisation=True, use_dropout=False):
    if use_batch_normalisation:
        disc_input = BatchNormalization()(disc_input)

    feature_vec = Flatten()(disc_input)
    dn_1 = Dense(30)(feature_vec)
    dn_1 = LeakyReLU(relu_param)(dn_1)

    dn_1 = Dense(30)(dn_1)
    dn_1 = LeakyReLU(relu_param)(dn_1)

    dn_1 = Dense(30)(dn_1)
    dn_1 = LeakyReLU(relu_param)(dn_1)

    dn_1 = Dense(10)(dn_1)
    dn_1 = LeakyReLU(relu_param)(dn_1)

    disc_out = Dense(1, activation='sigmoid')(dn_1)
    return disc_out

if __name__=='__main__':
    import sys
    sys.path.insert(0, '../../../')
    from Datamanager.OneD_Data_Creator import OneD_Data_Loader

    Data_loader = OneD_Data_Loader('Sqrs')
    # load Data ---------------------------------------------------------
    validation_split = 0.85

    lenss = 16
    xA_train = Data_loader.load_A()
    xB_train = Data_loader.load_B('B1')
    feats = Data_loader.load_A_Feat()

    xa_train_new = np.zeros((xA_train.shape[0], lenss, 1))
    xb_train_new = np.zeros((xA_train.shape[0], lenss, 1))

    for i in range(xA_train.shape[0]):
        xa_train_new[i, :, :] = xA_train[i, int(feats[i, 0,0]-8):int(feats[i, 0, 0]+8), :]
        xb_train_new[i, :, :] = xB_train[i, int(feats[i, 0,0]-8):int(feats[i, 0, 0]+8), :]

    xA_train = xa_train_new
    xB_train = xb_train_new

    print(xA_train.shape)
    print(xA_train[0, :, 0])
    print(xB_train[0, :, 0])


    xA_test = xA_train[0:int(xA_train.shape[0] * 0.1)]
    xB_test = xB_train[0:int(xB_train.shape[0] * 0.1)]

    xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
    xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

    xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
    xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
    # /load Data --------------------------------------------------------

    inp = Input(shape=(lenss, 1))#(int(xA_train[0].shape[0]/128), xA_train[0].shape[1]))
    d = disc_1D_full_dense(inp, 0.3, True, True)
    m = Model(inp, d, name='tst_disc_one_D')
    m.summary()
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    exit()

    all_data = np.zeros((xA_train.shape[0]+xB_train.shape[0], xA_train.shape[1], xA_train.shape[2]))
    all_data[0:xA_train.shape[0]] = xA_train
    all_data[xA_train.shape[0]:] = xB_train
    labels = np.zeros((xA_train.shape[0]+xB_train.shape[0],))
    labels[0:xA_train.shape[0]] = np.ones((xA_train.shape[0], ))

    shffle = np.arange(xA_train.shape[0] + xB_train.shape[0])
    np.random.shuffle(shffle)
    all_data = all_data[shffle]
    labels = labels[shffle]

    # validation_data
    all_valid_data = np.zeros(
        (xa_val.shape[0] + xb_val.shape[0], xa_val.shape[1], xa_val.shape[2]))
    all_valid_data[0:xa_val.shape[0]] = xa_val
    all_valid_data[xa_val.shape[0]:] = xb_val
    all_valid_labels = np.zeros((xa_val.shape[0] + xb_val.shape[0],))
    all_valid_labels[0:xa_val.shape[0]] = np.ones((xa_val.shape[0],))

    shffle_val = np.arange(xa_val.shape[0] + xb_val.shape[0])
    np.random.shuffle(shffle_val)
    all_valid_data = all_data[shffle_val]
    all_valid_labels = labels[shffle_val]


    m.fit(all_data, labels, validation_data=(all_valid_data, all_valid_labels), batch_size=128, epochs=50) #

    all_dataT = np.zeros((xA_test.shape[0] + xB_test.shape[0], xA_test.shape[1], xA_test.shape[2]))
    all_dataT[0:xA_test.shape[0]] = xA_test
    all_dataT[xA_test.shape[0]:] = xB_test
    labels_T = np.zeros((xA_test.shape[0] + xB_test.shape[0],))
    labels_T[0:xA_test.shape[0]] = np.ones((xA_test.shape[0],))
    test_score = m.evaluate(all_dataT, labels_T)

    print('Eval score: ',test_score)
    '''
    Epoch 50/50
    
      128/20000 [..............................] - ETA: 1s - loss: 1.7300e-07 - acc: 1.0000
     1024/20000 [>.............................] - ETA: 1s - loss: 1.1817e-06 - acc: 1.0000
     1920/20000 [=>............................] - ETA: 1s - loss: 8.7426e-07 - acc: 1.0000
     2688/20000 [===>..........................] - ETA: 1s - loss: 2.5415e-06 - acc: 1.0000
     3456/20000 [====>.........................] - ETA: 1s - loss: 2.1667e-06 - acc: 1.0000
     4224/20000 [=====>........................] - ETA: 1s - loss: 7.7634e-06 - acc: 1.0000
     4992/20000 [======>.......................] - ETA: 1s - loss: 7.0254e-06 - acc: 1.0000
     5888/20000 [=======>......................] - ETA: 0s - loss: 6.0789e-06 - acc: 1.0000
     6784/20000 [=========>....................] - ETA: 0s - loss: 5.3671e-06 - acc: 1.0000
     7424/20000 [==========>...................] - ETA: 0s - loss: 5.0573e-06 - acc: 1.0000
     8320/20000 [===========>..................] - ETA: 0s - loss: 4.6138e-06 - acc: 1.0000
     9216/20000 [============>.................] - ETA: 0s - loss: 4.7438e-06 - acc: 1.0000
     9984/20000 [=============>................] - ETA: 0s - loss: 4.4455e-06 - acc: 1.0000
    10752/20000 [===============>..............] - ETA: 0s - loss: 4.2251e-06 - acc: 1.0000
    11520/20000 [================>.............] - ETA: 0s - loss: 3.9801e-06 - acc: 1.0000
    12416/20000 [=================>............] - ETA: 0s - loss: 3.8072e-06 - acc: 1.0000
    13056/20000 [==================>...........] - ETA: 0s - loss: 3.6381e-06 - acc: 1.0000
    13952/20000 [===================>..........] - ETA: 0s - loss: 5.3477e-06 - acc: 1.0000
    14720/20000 [=====================>........] - ETA: 0s - loss: 5.6063e-06 - acc: 1.0000
    15488/20000 [======================>.......] - ETA: 0s - loss: 5.4311e-06 - acc: 1.0000
    16256/20000 [=======================>......] - ETA: 0s - loss: 5.2563e-06 - acc: 1.0000
    17024/20000 [========================>.....] - ETA: 0s - loss: 5.2022e-06 - acc: 1.0000
    17664/20000 [=========================>....] - ETA: 0s - loss: 5.0553e-06 - acc: 1.0000
    18560/20000 [==========================>...] - ETA: 0s - loss: 4.8377e-06 - acc: 1.0000
    19456/20000 [============================>.] - ETA: 0s - loss: 4.9112e-06 - acc: 1.0000
    20000/20000 [==============================] - 1s 72us/step - loss: 4.8150e-06 - acc: 1.0000 - val_loss: 1.2232e-05 - val_acc: 1.0000
    
     32/300 [==>...........................] - ETA: 0s
    300/300 [==============================] - 0s 67us/step
    Eval score:  [7.194075658105703e-06, 1.0]
    '''


