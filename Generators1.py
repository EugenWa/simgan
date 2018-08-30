# - keras -
import keras
from keras.layers import BatchNormalization, Conv2D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling2D, \
    UpSampling2D, Concatenate
from keras.models import Model
CUDA_VISIBLE_DEVICES=0


def scaling(x):
    return x / 255


def upscaling(x):
    return x * 255


def scaling_shape(input_shape):
    return input_shape


Scaling = Lambda(scaling, output_shape=scaling_shape, name='Scaling')
UpScaling = Lambda(upscaling, output_shape=scaling_shape, name='UpScaling')
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------

'''
    This generator will keep the image dimensions in each Conv-layer due to the idea that
    the 'simulated image' (aka perfect, wihtout noise) is supposed to be a valid feature
    representation of the data, implying that there might be no need for reducing the 
    feature vector any further

    Notes: this generator will use the leaky-relu's instead of relus to ensure
            a smooth mapping function
'''


def build_generator_conserving_ImgDim_basic(generator_input, generator_name, use_batch_normalisation=True):
    scaled_input = generator_input  # Scaling(generator_input)
    if use_batch_normalisation:
        scaled_input = BatchNormalization()(scaled_input)

    # use 4 Convolution layers to transform the input
    c_1 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(scaled_input)
    c_1 = LeakyReLU(0.2)(c_1)

    c_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_1)
    c_2 = LeakyReLU(0.2)(c_2)
    if use_batch_normalisation:
        c_2 = BatchNormalization()(c_2)

    c_3 = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_2)
    c_3 = LeakyReLU(0.2)(c_3)

    gen_output = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_3)
    ##gen_output = UpScaling(gen_output)

    gen_Model = Model(inputs=[generator_input], outputs=[gen_output], name=generator_name)
    gen_Model.summary()

    return gen_Model


'''
    This will be the same type of Generator as the basic, conserving the original Image dimensions,
    but with one residual Blocks in between
'''


def build_generator_conserving_ImgDim_one_ResBlock(generator_input, generator_name, use_batch_normalisation=True):
    scaled_input = generator_input  # Scaling(generator_input)
    if use_batch_normalisation:
        scaled_input = BatchNormalization()(scaled_input)

    c_1 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(scaled_input)
    c_1 = LeakyReLU(0.2)(c_1)

    c_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_1)
    c_2 = LeakyReLU(0.2)(c_2)
    if use_batch_normalisation:
        c_2 = BatchNormalization()(c_2)

    # Add two residual Blocks
    r_1 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_2)
    r_1 = LeakyReLU(0.2)(r_1)
    r_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(r_1)
    r_2 = LeakyReLU(0.2)(r_2)
    if use_batch_normalisation:
        r_2 = BatchNormalization()(r_2)

    # concatenate the residuum
    con_1 = Concatenate()([c_2, r_2])
    c_3 = Conv2D(9, kernel_size=(1, 1), strides=(1, 1), padding='same')(con_1)

    c_4 = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_3)
    c_4 = LeakyReLU(0.2)(c_4)

    gen_output = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_4)
    # gen_output = UpScaling(gen_output)

    gen_Model = Model(inputs=[generator_input], outputs=[gen_output], name=generator_name)
    gen_Model.summary()

    return gen_Model


'''
    This will be the same type of Generator as the basic, conserving the original Image dimensions,
    but with two residual Blocks in between, so it add one additional 'layer' of complexity
'''


def build_generator_conserving_ImgDim_two_ResBlock(generator_input, generator_name, use_batch_normalisation=True):
    scaled_input = generator_input  # Scaling(generator_input)
    if use_batch_normalisation:
        scaled_input = BatchNormalization()(scaled_input)

    c_1 = Conv2D(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False)(scaled_input)
    c_1 = LeakyReLU(0.2)(c_1)

    c_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_1)
    c_2 = LeakyReLU(0.2)(c_2)
    if use_batch_normalisation:
        c_2 = BatchNormalization()(c_2)

    # Add two residual Blocks
    r_1 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_2)
    r_1 = LeakyReLU(0.2)(r_1)
    r_2 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(r_1)
    r_2 = LeakyReLU(0.2)(r_2)
    if use_batch_normalisation:
        r_2 = BatchNormalization()(r_2)

    # concatenate the residuum
    con_1 = Concatenate()([c_2, r_2])
    c_3 = Conv2D(9, kernel_size=(1, 1), strides=(1, 1), padding='same')(con_1)

    # Add two residual Blocks
    r_3 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_3)
    r_3 = LeakyReLU(0.2)(r_3)
    r_4 = Conv2D(9, kernel_size=(3, 3), strides=(1, 1), padding='same')(r_3)
    r_4 = LeakyReLU(0.2)(r_4)
    if use_batch_normalisation:
        r_4 = BatchNormalization()(r_4)

    # concatenate the residuum
    con_2 = Concatenate()([c_3, r_4])
    c_4 = Conv2D(9, kernel_size=(1, 1), strides=(1, 1), padding='same')(con_2)

    c_5 = Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_4)
    c_5 = LeakyReLU(0.2)(c_5)

    gen_output = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(c_5)

    # gen_output = UpScaling(gen_output)

    gen_Model = Model(inputs=[generator_input], outputs=[gen_output], name=generator_name)
    gen_Model.summary()

    return gen_Model
