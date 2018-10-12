# - keras -
import keras
from keras.layers import BatchNormalization, Input, LeakyReLU, Add, Lambda, MaxPooling1D, \
    UpSampling1D, Concatenate, Dropout, Conv1D
from keras.models import Model, load_model
import numpy as np
CUDA_VISIBLE_DEVICES=0


def build_basic_encoder_residual(encoder_input, e_filters=[3, 9, 21, 33], connectivity_layers=[0, 1, 2, 3], relu_param=0.3, use_batch_normalisation=True, use_dropout=False):
    if use_batch_normalisation:
        encoder_input = BatchNormalization()(encoder_input)

    skipping_cons = []
    filter_len=3
    c_L = encoder_input
    for i in range(len(e_filters)):

        if i in connectivity_layers:
            skipping_cons.append([c_L, keras.backend.int_shape(c_L)[-1]])

        c_L = Conv1D(e_filters[i], kernel_size=filter_len, strides=1, padding='same')(c_L)
        c_L = LeakyReLU(relu_param)(c_L)
        c_L = MaxPooling1D()(c_L)

        if use_batch_normalisation and (i%2 is 1):
            c_L = BatchNormalization()(c_L)
        if use_dropout:
            c_L = Dropout(0.2)(c_L)

    encoder_out = c_L

    return encoder_out, skipping_cons


def build_basic_decoder_residual(decoder_input, skipping_cons, build_res, d_filters=[33, 21, 9, 3], connectivity_layers=[0, 1, 2, 3], relu_param=0.3, use_batch_normalisation=True, use_dropout=False):
    if use_batch_normalisation:
        decoder_input = BatchNormalization()(decoder_input)

    connectivity_counter = 0
    filter_len = 3
    c_L = decoder_input
    # create encoding blocks
    for i in range(len(d_filters)):
        c_L = Conv1D(d_filters[i], kernel_size=filter_len, strides=1, padding='same', name='convsss'+str(i))(c_L)
        c_L = LeakyReLU(relu_param)(c_L)
        c_L = UpSampling1D()(c_L)

        if use_batch_normalisation and (i%2 is 1) and (i < len(d_filters)-1):
            c_L = BatchNormalization()(c_L)
        if use_dropout and (i < len(d_filters)-1):
            c_L = Dropout(0.2)(c_L)
        if build_res:
            if i in connectivity_layers:
                con_cat = Concatenate()([c_L, skipping_cons[connectivity_counter][0]])
                c_L = Conv1D(skipping_cons[connectivity_counter][1], kernel_size=1, strides=1, padding='same')(con_cat)
                connectivity_counter += 1



    dec_output = Conv1D(1, kernel_size=3, strides=1, padding='same', name='end_conv')(c_L)     # add activatoin sigmoid
    dec_output = LeakyReLU(relu_param)(dec_output)

    return dec_output


def build_basic_transformation_layers(transform_input, layer_amount, layer_size=33, relu_param=0.3, use_batch_normalisation=True, use_dropout=False):
    block_in = transform_input
    filter_len = 3
    for i in range(layer_amount):
        r_1 = Conv1D(layer_size, kernel_size=filter_len, strides=1, padding='same')(block_in)
        r_1 = LeakyReLU(relu_param)(r_1)
        r_2 = Conv1D(layer_size, kernel_size=filter_len, strides=1, padding='same')(r_1)
        r_2 = LeakyReLU(relu_param)(r_2)
        if use_batch_normalisation:
            r_2 = BatchNormalization()(r_2)
        if use_dropout and ((i%2 is 1) or (0 < i and (i is len(layer_amount)-2))):
            r_2 = Dropout(0.2)(r_2)

        # concatenate the residuum
        con_1 = Concatenate()([block_in, r_2])
        block_in = Conv1D(layer_size, kernel_size=1, strides=1, padding='same')(con_1)

    return block_in

# testing
if __name__=='__main__':
    inp = Input(shape=(128, 1))

    ft, skippings = build_basic_encoder_residual(inp)
    tr = build_basic_transformation_layers(ft, 2, 33)
    out = build_basic_decoder_residual(tr, list(reversed(skippings)), True)
    out2 = build_basic_decoder_residual(tr, list(reversed(skippings)), False)

    m = Model(inp, out)
    m.summary()
    m.compile(optimizer='adam', loss='mae')

    m2 = Model(inp, out2)
    m2.summary()
    m2.compile(optimizer='adam', loss='mae')

    keras.utils.plot_model(m, 'tst.png')
    keras.utils.plot_model(m2, 'tst2.png')



