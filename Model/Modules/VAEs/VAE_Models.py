# import default path
import os
import sys
sys.path.insert(0, '../../../')

from Model.Modules.Generators.Generator_Blocks import build_transformation_layer_basic, build_decoder_basic, build_encoder_basic
# - keras -
import keras
from keras.layers import Input
from keras.models import Model, load_model
import numpy as np



class Basic_VAE:
    def __init__(self, vae_name, img_shape, cfg, vid='', path=None):
        self.name = vae_name
        self.optimizers={'adam':keras.optimizers.adam}

        if path is None:
            vae_input_id = Input(shape=img_shape)
            vae_input_no = Input(shape=img_shape)

            lr          = cfg['VAE' + vid]['LEARNING_RATE']
            lr_decay    = cfg['VAE' + vid]['LR_DEF']
            relu_param  = cfg['VAE' + vid]['RELU_PARAM']
            optimizer   = self.optimizers[cfg['VAE' + vid]['OPTIMIZER']]
            use_drop_out            = cfg['VAE' + vid]['USE_DROP_OUT']
            use_batch_normalisation = cfg['VAE' + vid]['USE_BATCH_NORM']
            trafo_layers            = cfg['VAE' + vid]['TRAFO_LAYERS']
            filters     = cfg['VAE' + vid]['FILTERS']
            filters_rev = list(reversed(filters))
            feature_inp = Input(shape=(img_shape[0]/(2**len(filters)), img_shape[1]/(2**len(filters)), filters[-1]))


            # -- VAE Model vid ---
            encoder_ID_features = build_encoder_basic(vae_input_id, filters, relu_param, use_batch_normalisation, use_drop_out)
            decoder_ID_out      = build_decoder_basic(feature_inp, filters_rev, relu_param, use_batch_normalisation, use_drop_out)
            self.decoder_optimizer  = optimizer(lr, lr_decay)
            self.Decoder            = Model(inputs=[feature_inp], outputs=[decoder_ID_out], name=vae_name + '_decoder')

            print('Decoder Layout: ')
            self.Decoder.summary()
            self.Decoder.compile(optimizer=self.decoder_optimizer, loss=cfg['VAE' + vid]['DECODER_LOSS'])

            # build vae_id encoder:
            vae__out_id = self.Decoder(encoder_ID_features)
            self.vae_id_optimizer   = optimizer(lr, lr_decay)
            self.VAE_ID             = Model(inputs=[vae_input_id], outputs=[vae__out_id], name=vae_name + '_vaeID')
            self.VAE_ID_MultiOut    = Model(inputs=[vae_input_id], outputs=[vae__out_id, encoder_ID_features], name=vae_name + '_vaeIDMOU')

            print('VAE-ID Layout: ')
            self.VAE_ID.summary()
            self.VAE_ID_MultiOut.compile(optimizer=self.vae_id_optimizer, loss=[cfg['VAE' + vid]['IMAGE_LOSS'], None], loss_weights=[1, None])
            self.VAE_ID.compile(optimizer=self.vae_id_optimizer, loss=cfg['VAE' + vid]['IMAGE_LOSS'])

            # freeze the decoder in the domain A->B setting hence it should learn to reconstruct the features
            self.Decoder.trainable=False

            # transformation Model
            encoder_NO_features = build_encoder_basic(vae_input_no, filters, relu_param, use_batch_normalisation, use_drop_out)
            transfo_NO_features = build_transformation_layer_basic(encoder_NO_features, trafo_layers, filters[-1], relu_param, use_batch_normalisation, use_drop_out)
            vae__out_no         = self.Decoder(transfo_NO_features)
            self.vae_no_optimizer   = optimizer(lr, lr_decay)
            self.VAE_NO             = Model(inputs=[vae_input_no], outputs=[vae__out_no, transfo_NO_features], name=vae_name + '_vaeNO')
            self.VAE_NO_MulitOut    = Model(inputs=[vae_input_no], outputs=[vae__out_no, transfo_NO_features, encoder_NO_features], name=vae_name + '_vaeNOMOU')

            print('VAE-NO Layout: ')
            self.VAE_NO.summary()
            self.VAE_NO.compile(optimizer=self.vae_no_optimizer, loss=[cfg['VAE' + vid]['IMAGE_LOSS'], cfg['VAE' + vid]['FEATURE_LOSS']], loss_weights=cfg['VAE' + vid]['LOSS_WEIGHTS'])
            self.VAE_NO_MulitOut.compile(optimizer=self.vae_no_optimizer, loss=[cfg['VAE' + vid]['IMAGE_LOSS'], cfg['VAE' + vid]['FEATURE_LOSS'], None], loss_weights=[1, 1, None])
            # lossweights unimportant hence this moel wont be trained
        else:
            self.load_Model(path)


    def train_model_on_batch(self, xA_Batch, xB_Batch, vid_ONLY=False):
        loss_id = self.VAE_ID.train_on_batch([xB_Batch], [xB_Batch])
        if vid_ONLY:
            return loss_id, loss_id, None
        # predict features
        _, features = self.VAE_ID_MultiOut.predict(xB_Batch)
        loss_map = self.VAE_NO.train_on_batch([xA_Batch], [xB_Batch, features])
        return np.add(loss_id, loss_map)*0.5, loss_id, loss_map

    def predict_ID(self, batch):
        return self.VAE_ID_MultiOut.predict(batch)

    def predict_NO(self, batch):
        return self.VAE_NO_MulitOut.predict(batch)

    def predict_ID_img_only(self, batch):
        return self.VAE_ID.predict(batch)

    def predict_NO_img_only(self, batch):
        img, _ = self.VAE_NO.predict(batch)
        return img

    def save_model(self, path, obj=''):
        self.Decoder.save(path + '/decoder' + obj + '.h5')
        self.VAE_ID.save(path + '/vaeID' + obj+ '.h5')
        self.VAE_NO.save(path + '/vaeNO' + obj+ '.h5')
        self.VAE_ID_MultiOut.save(path + '/vaeIDMOU'+ obj + '.h5')
        self.VAE_NO_MulitOut.save(path + '/vaeNOMOU'+ obj + '.h5')

    def load_Model(self, path, obj=''):
        if self.Decoder is not None:
            del self.Decoder
        if self.VAE_ID is not None:
            del self.VAE_ID
        if self.VAE_NO is not None:
            del self.VAE_NO
        if self.VAE_NO_MulitOut is not None:
            del self.VAE_NO_MulitOut
        if self.VAE_ID_MultiOut is not None:
            del self.VAE_ID_MultiOut

        #keras.backend.clear_session()

        self.Decoder = load_model(path + '/decoder' + obj+ '.h5')
        self.VAE_ID = load_model(path + '/vaeID'+ obj+ '.h5')
        self.VAE_NO = load_model(path + '/vaeNO'+ obj+ '.h5')
        self.VAE_ID_MultiOut = load_model(path + '/vaeIDMOU'+ obj+ '.h5')
        self.VAE_NO_MulitOut = load_model(path + '/vaeNOMOU'+ obj+ '.h5')

