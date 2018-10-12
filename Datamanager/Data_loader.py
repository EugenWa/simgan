import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from glob import glob
import os

class Data_Loader:
    def __init__(self, data_set_name):
        self.data_set_name = data_set_name

    '''
        Loads two Tensors of Data, scaled from 0.0 to 1.0, it loads them as they are in the filesystem -> no shuffling
        data_type € {'train', 'test'}
    '''
    def Load_Data_Tensors(self, data_type, invert=True):
        Datasetpath = os.path.dirname(os.path.abspath(__file__))
        Datasetpath = Datasetpath[0:-(len("Datamanager") + 1)]

        paths_A = glob(Datasetpath + '/Datasets/%s/%sA/*' % (self.data_set_name, data_type))
        paths_B = glob(Datasetpath + '/Datasets/%s/%sB/*' % (self.data_set_name, data_type))

        paths_A = sorted(paths_A)
        paths_B = sorted(paths_B)
        img = plt.imread(paths_A[0])
        # crop the image so that the final shape is a square
        img_shapemin = min(img.shape[0], img.shape[1])

        # scale so that filters can be fit:
        number_of_divisions = 5
        while img_shapemin % (2 ** number_of_divisions) != 0:
            img_shapemin -= 1
        # create data tensors for images from domain A and B
        PA_data = np.zeros((len(paths_A), img_shapemin, img_shapemin, 1))
        PB_data = np.zeros((len(paths_B), img_shapemin, img_shapemin, 1))

        # read images into matrix
        for i in range(len(paths_A)):
            img = plt.imread(paths_A[i])
            # crop
            img = img[0:img_shapemin, 0:img_shapemin]
            if invert:
                img = img*(-1)+1
            img = np.expand_dims(img, axis=3)
            PA_data[i] = img

        for i in range(len(paths_B)):
            img = plt.imread(paths_B[i])
            # crop
            img = img[0:img_shapemin, 0:img_shapemin]
            if invert:
                img = img * (-1) + 1
            img = np.expand_dims(img, axis=3)
            PB_data[i] = img

        return PA_data, PB_data

    def Load_Data_Tensor_Split(self, data_type, validation_split, test_split=0.1, invert=True):
        # load Data --------------------------------------------------------
        xA_train, xB_train = self.Load_Data_Tensors(data_type, invert)

        xA_test = xA_train[0:int(xA_train.shape[0] * test_split)]
        xB_test = xB_train[0:int(xB_train.shape[0] * test_split)]

        xA_train = xA_train[int(xA_train.shape[0] * test_split):]
        xB_train = xB_train[int(xB_train.shape[0] * test_split):]

        xa_val = xA_test[0:int(xA_test.shape[0] * validation_split)]
        xb_val = xB_test[0:int(xB_test.shape[0] * validation_split)]

        xA_test = xA_test[int(xA_test.shape[0] * validation_split):]
        xB_test = xB_test[int(xB_test.shape[0] * validation_split):]
        # /load Data --------------------------------------------------------

        xA_train_full = xA_train
        xA_val_full   = xa_val
        xA_test_full  = xA_test

        return xA_train, xA_train_full, xB_train, xa_val, xA_val_full, xb_val, xA_test, xA_test_full, xB_test

