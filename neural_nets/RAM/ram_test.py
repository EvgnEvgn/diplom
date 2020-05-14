from neural_nets.RAM.RecurrentAttentionModel import RecurrentAttentionModel
import tensorflow.compat.v1 as tf
from neural_nets.RAM.data_loader import read_data_sets
from neural_nets.RAM.Dataset import Dataset
from neural_nets.RAM.ram_config import RAM_Config
import pickle
import numpy as np

tf.disable_v2_behavior()


def get_test_tchgn_dgts_dataset(filename):
    with open(filename, "rb") as file:
        dataset = pickle.load(file)

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]

    return test_X, test_Y


def get_tchng_dgts_small_dataset(filename):
    with open(filename, "rb") as file:
        dataset = pickle.load(file)

    train_X = dataset["train_X"]
    train_Y = dataset["train_Y"]

    val_X = dataset["val_X"]
    val_Y = dataset["val_Y"]

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


#train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_data_sets("mnist_data")

#test_X, test_Y = get_test_tchgn_dgts_dataset("../../testtc")
#test_X, test_Y = get_test_tchgn_dgts_dataset("../../test_date_numbers_75examples_alt.bin")
#test_X = np.reshape(test_X, (test_X.shape[0], 4096))
#test_Y = np.array(test_Y, dtype=np.int64)
#test_Y = test_Y % 10
# indxs = np.where(test_Y > 9)[0]
# test_X = test_X[indxs]
# test_Y = test_Y[indxs]
#test_Y -= 10
#test_X = test_X[0:5000]
#test_Y = test_Y[0:5000]
batch_size = 24

#test_Y -= 10
train_dataset = Dataset(train_images, train_labels, batch_size=batch_size, reshape_img=True)
val_dataset = Dataset(validation_images, validation_labels, batch_size=batch_size, reshape_img=True)
# #temp_dataset = Dataset(val_X, val_Y)
test_dataset = Dataset(test_images, test_labels, batch_size=batch_size, reshape_img=True)
# t123_dataset = Dataset(val_X[0:10], val_Y[0:10], batch_size=batch_size)
config = RAM_Config(batch_size=batch_size, max_epochs=10, num_classes=10, nGlimpses=6, translated=False)

ram = RecurrentAttentionModel(config)
ram.train(train_dataset, val_dataset)
#ram.predict(test_dataset, batch_size=batch_size, draw=False)
