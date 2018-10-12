# - keras -
import keras
from keras.layers import BatchNormalization, Input, LeakyReLU, Add, Lambda, MaxPooling1D, \
    UpSampling1D, Concatenate, Dropout, Conv1D, Flatten, Dense


def build_feature_classifier(encoder_feat, e_filters=[33, 10, 3], denses=[20, 20], relu_param=0.3, use_batch_normalisation=True, use_dropout=False):
    if use_batch_normalisation:
        encoder_feat = BatchNormalization()(encoder_feat)

    filter_len=3
    c_L = encoder_feat
    for i in range(len(e_filters)):
        c_L = Conv1D(e_filters[i], kernel_size=filter_len, strides=1, padding='same')(c_L)
        c_L = LeakyReLU(relu_param)(c_L)

        if use_batch_normalisation and (i%2 is 1):
            c_L = BatchNormalization()(c_L)
        if use_dropout:
            c_L = Dropout(0.2)(c_L)

    c_L = Flatten()(c_L)

    for i in range(len(denses)):
        c_L = Dense(denses[i])(c_L)
        c_L = LeakyReLU(relu_param)(c_L)

    encoder_out = c_L

    return encoder_out






