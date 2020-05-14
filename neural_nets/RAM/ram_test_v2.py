from neural_nets.RAM.RecurrentAttentionModel_v2 import RecurrentAttentionModel_v2 as RecurrentAttentionModel
import tensorflow.compat.v1 as tf
from neural_nets.RAM.data_loader import read_data_sets
from neural_nets.RAM.Dataset import Dataset
from neural_nets.RAM.ram_config_v2 import RAM_Config
import pickle
import numpy as np
import cv2

tf.disable_v2_behavior()


def get_test_tchgn_dgts_dataset(filename):
    with open(filename, "rb") as file:
        dataset = pickle.load(file)

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]

    return test_X, test_Y


def resize_imgs(data):
    #reshaped = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2]))
    resized = np.array([cv2.resize(img, (52, 28)) for img in data])

    return resized


def get_tchng_dgts_small_dataset(filename):
    with open(filename, "rb") as file:
        dataset = pickle.load(file)

    train_X = dataset["train_X"]
    train_Y = dataset["train_Y"]

    val_X = dataset["val_X"]
    val_Y = dataset["val_Y"]

    # test_X = dataset["test_X"]
    # test_Y = dataset["test_Y"]

    return train_X, train_Y, val_X, val_Y


# train_images, train_labels, validation_images, validation_labels, test_images, test_labels = read_data_sets(
#     "mnist_data")
# train_images = resize_imgs(train_images)
# validation_images = resize_imgs(validation_images)
# test_images = resize_imgs(test_images)


test_X, test_Y, val_X, val_Y = get_tchng_dgts_small_dataset("../../train_dataset_rus_text_score.bin")
# test_X, test_Y = get_test_tchgn_dgts_dataset("../../test_date_numbers_75examples_alt.bin")
# test_X = np.reshape(test_X, (test_X.shape[0], 4096))
# test_Y = np.array(test_Y, dtype=np.int64)
# test_Y = test_Y % 10
# indxs = np.where(test_Y > 9)[0]
# test_X = test_X[indxs]
# test_Y = test_Y[indxs]
# test_Y -= 10
# test_X = test_X[0:5000]
# test_Y = test_Y[0:5000]
batch_size = 20

# test_Y -= 10
train_dataset = Dataset(test_X, test_Y, batch_size=batch_size, reshape_img=False)
val_dataset = Dataset(val_X, val_Y, batch_size=batch_size, reshape_img=False)
# #temp_dataset = Dataset(val_X, val_Y)
#test_dataset = Dataset(test_images, test_labels, batch_size=batch_size, reshape_img=True)
# t123_dataset = Dataset(val_X[0:10], val_Y[0:10], batch_size=batch_size)
config = RAM_Config(batch_size=batch_size, max_epochs=200, num_classes=10, nGlimpses=6, translated=False)

ram = RecurrentAttentionModel(config)
ram.train(train_dataset, val_dataset)
#ram.predict(val_dataset, batch_size=batch_size, draw=True)
