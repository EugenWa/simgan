import os
import sys
import pickle
import numpy as np
sys.path.insert(0, '../../../')
# - keras -
import keras
from keras.layers import Input, LeakyReLU, Add, Lambda, Concatenate, Dense, Flatten, Reshape, Conv1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from Model.Modules.Generators.Generator_Basic import build_basic_decoder, build_basic_encoder, build_basic_transformation_layers
from Model.Modules.Static_Objects import possible_loss_functions, possible_optimizers
from Model.Modules.Generators.Feature_Classifier import build_feature_classifier


class Basic_nRES_AE_2D:
    def __init__(self, ae_name, img_shape, cfg, AE_eval):
        self.name = ae_name
        self.optimizers = possible_optimizers           # {'adam':keras.optimizers.adam ......}
        self.loss_functions = possible_loss_functions   # {'mae':keras.losses.mae, 'mse':keras.losses.mse ......}
        self.ae_eval = AE_eval
        self.seed = cfg['SEED']

        lr              = cfg['VAE']['LEARNING_RATE']
        lr_decay        = cfg['VAE']['LR_DEF']
        relu_param      = cfg['VAE']['RELU_PARAM']
        optimizer       = self.optimizers[cfg['VAE']['OPTIMIZER']]
        use_drop_out    = cfg['VAE']['USE_DROP_OUT']
        use_batch_normalisation = cfg['VAE']['USE_BATCH_NORM']
        trafo_layers            = cfg['VAE']['TRAFO_LAYERS']
        filters                 = cfg['VAE']['FILTERS']
        metric_2_be_used        = cfg['METRIC']
        classify_mode           = cfg['FEATURE_CLASSIFY']
        CLASSIF_FILTERS         = cfg['VAE']['CLASSIF_FILTERS']
        CLASSIF_DENSES          = cfg['VAE']['CLASSIF_DENSES']
        CLASSIF_LOSS            = cfg['VAE']['CLASSIF_LOSS']


        inp = Input(img_shape)
        feat = build_basic_encoder(inp, filters, relu_param, use_batch_normalisation, use_drop_out)
        if trafo_layers > 0:
            tfeat = build_basic_transformation_layers(feat, trafo_layers, filters[-1], relu_param, use_batch_normalisation, use_drop_out)
        else:
            tfeat = feat
        out = build_basic_decoder(tfeat, list(reversed(filters)), relu_param, use_batch_normalisation, use_drop_out)

        # compile model
        ae_optimizer = optimizer(lr, lr_decay)
        if metric_2_be_used is 1:
            print('using trim metric')
            def ae_loss_function(xx, yy):
                yy_xx_abs = keras.backend.abs(yy[:, :, 0]-xx[:, :, 0])
                c_cond = keras.backend.less(yy_xx_abs, 0.005)
                a_new  = keras.backend.switch(c_cond, keras.backend.zeros_like(yy_xx_abs), yy_xx_abs)
                return keras.backend.mean(a_new)

            autoencoder_loss = ae_loss_function
        else:
            autoencoder_loss = cfg['VAE']['IMAGE_LOSS']


        self.auto_encoder = Model(inp, out, name=ae_name)
        self.auto_encoder.compile(optimizer=ae_optimizer, loss=autoencoder_loss)

        print('Basic nRes AE 2D: ')
        self.auto_encoder.summary()
        # plot model to file
        keras.utils.plot_model(self.auto_encoder, self.ae_eval.model_saves_dir + '/' + self.auto_encoder.name + '.png')

    def Call(self):
        return self.auto_encoder

    def load_pretrained_model(self, Type):
        if Type=='ID':
            try:
                self.auto_encoder.load_weights(self.ae_eval.model_saves_dir + '/AE_ID.h5')
            except Exception:
                print('NO ID MODEL')
        else:
            try:
                self.auto_encoder.load_weights(self.ae_eval.model_saves_dir + '/AE_NO.h5')
            except Exception:
                print('NO NO MODEL')

    def fit_ID(self, xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only):
        np.random.seed(self.seed)  # equal shuffling
        check_pointer_ID = ModelCheckpoint(self.ae_eval.model_saves_dir + '/AE_ID.h5', verbose=verbose, save_best_only=save_best_only)
        history_id = self.auto_encoder.fit(xB_train, xA_train, validation_data=(xB_eval, xA_eval), shuffle=shuffle, epochs=epochs, batch_size=batch_size, callbacks=[check_pointer_ID])

        self.auto_encoder.load_weights(self.ae_eval.model_saves_dir + '/AE_ID.h5')
        eval_score_id_best = self.auto_encoder.evaluate(xB_test, xA_test, batch_size=batch_size)
        print('Best ID-Model score: ', eval_score_id_best)
        print('----------------------------------------------------------------------')

        # save histories
        try:
            with open(self.ae_eval.training_history_dir + '/Hist_ID', "rb") as fp:
                history_id_previous = pickle.load(fp)
        except Exception:
            history_id_previous = {'loss': [], 'val_loss': []}

        # append new found data
        history_id_previous['loss'].extend(history_id.history['loss'])
        history_id_previous['val_loss'].extend(history_id.history['val_loss'])

        # safe history
        with open(self.ae_eval.training_history_dir + '/Hist_ID', "wb") as fp:
            pickle.dump(history_id_previous, fp)

    def fit_NO(self, xA_train, xB_train, xA_eval, xB_eval, xA_test, xB_test, epochs, batch_size, shuffle, verbose, save_best_only):
        np.random.seed(self.seed)  # equal shuffling
        check_pointer_NO = ModelCheckpoint(self.ae_eval.model_saves_dir + '/AE_NO.h5', verbose=verbose, save_best_only=save_best_only)
        history_no = self.auto_encoder.fit(xB_train, xA_train, validation_data=(xB_eval, xA_eval), shuffle=shuffle, epochs=epochs, batch_size=batch_size, callbacks=[check_pointer_NO])

        self.auto_encoder.load_weights(self.ae_eval.model_saves_dir + '/AE_NO.h5')
        eval_score_no_best = self.auto_encoder.evaluate(xB_test, xA_test, batch_size=batch_size)
        print('Best NO-Model score: ', eval_score_no_best)
        print('----------------------------------------------------------------------')

        # save histories
        try:
            with open(self.ae_eval.training_history_dir + '/Hist_NO', "rb") as fp:
                history_no_previous = pickle.load(fp)
        except Exception:
            history_no_previous = {'loss': [], 'val_loss': []}

        # append new found data
        history_no_previous['loss'].extend(history_no.history['loss'])
        history_no_previous['val_loss'].extend(history_no.history['val_loss'])

        # safe training history
        with open(self.ae_eval.training_history_dir + '/Hist_NO', "wb") as fp:
            pickle.dump(history_no_previous, fp)

    def train_on_batch(self, xA_batch, xB_batch):
        return self.auto_encoder.train_on_batch(xB_batch, xA_batch)

