import sys
import platform
import numpy as np
import scipy.ndimage.interpolation

sys.path.append('../')
from neural_nets.RAM_V2.mnist import MNISTData

DATA_PATH = '../RAM/mnist_data/'


def original_mnist(batch_size=128, shuffle=True):
    def preprocess_im(im):
        """ normalize input image to [0., 1.] """
        im = im.astype(np.float32)
        im = im / 255.
        return im

    train_data = MNISTData('train', data_dir=DATA_PATH, shuffle=shuffle,
                           batch_dict_name=['data', 'label'], pf=preprocess_im)
    train_data.setup(epoch_val=0, batch_size=batch_size)
    valid_data = MNISTData('val', data_dir=DATA_PATH, shuffle=shuffle,
                           batch_dict_name=['data', 'label'], pf=preprocess_im)
    valid_data.setup(epoch_val=0, batch_size=batch_size)
    return train_data, valid_data
