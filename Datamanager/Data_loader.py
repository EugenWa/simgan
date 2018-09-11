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
