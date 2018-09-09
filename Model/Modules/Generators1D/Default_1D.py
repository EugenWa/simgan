import os
import sys
sys.path.insert(0, '../../../')
# - keras -
import keras
from keras.layers import Conv2D, Input, Conv2DTranspose, LeakyReLU, Add, Lambda, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model
from Model.Modules.Generators1D.Generator_Basic import build_basic_decoder, build_basic_encoder, build_basic_transformation_layers

CUDA_VISIBLE_DEVICES=0


def VAE_RES(vae_name, img_shape, filters, relu_param=0.3, use_batch_norm=True, use_dropout=False):
    inp = Input(img_shape)

    feat = build_basic_encoder(inp, filters,relu_param, use_batch_norm, use_dropout)
    tfeat = build_basic_transformation_layers(feat, 2,filters[-1],relu_param, use_batch_norm, use_dropout)
    out = build_basic_decoder(tfeat, list(reversed(filters)),relu_param, use_batch_norm, use_dropout)

    m = Model(inp, out, name=vae_name)

    print('Default VAE: ')
    m.summary()

    return m

def VAE_NO_RES(vae_name, img_shape, filters, relu_param = 0.3, use_batch_norm=True, use_dropout=False):
    inp = Input(img_shape)

    feat = build_basic_encoder(inp, filters, relu_param, use_batch_norm, use_dropout)
    out = build_basic_decoder(feat, list(reversed(filters)), relu_param, use_batch_norm, use_dropout)

    m = Model(inp, out, name=vae_name)

    print('Default VAE: ')
    m.summary()

    return m


