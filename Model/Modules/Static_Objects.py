import sys

sys.path.insert(0, '../../')
import keras

possible_loss_functions = {'mae':keras.losses.mae, 'mse':keras.losses.mse}
possible_optimizers     = {'adam':keras.optimizers.adam}