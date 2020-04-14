import mnist
# from neural_nets.features_utils import hog_feature, extract_features
import numpy as np
import emnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import config
import pandas as pd
import cv2
import os
import codecs
import pickle
import h5py
from sklearn.preprocessing import LabelEncoder


def get_emnist_letters_data_TF(num_training=122000, num_validation=2800, num_test=20800):
    x_train, y_train = emnist.extract_training_samples('letters')
    x_test, y_test = emnist.extract_test_samples('letters')
    # Переход к меткам диапазона [0..25]
    y_train = np.subtract(y_train, 1)
    y_test = np.subtract(y_test, 1)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.05)
    x_train = np.divide(x_train, 255).astype("float64")
    x_val = np.divide(x_val, 255).astype("float64")
    x_test = np.divide(x_test, 255).astype("float64")

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_letters_img_for_predictive_model(letters, model_input_shape):
    letters = np.divide(letters, 255).astype("float64")

    letters_reshaped = letters.reshape(model_input_shape)

    return letters_reshaped


def preprocess_digits_img_for_tf_predictive_model(digits, model_input_shape):
    digits = np.divide(digits, 255).astype("float64")

    digits = digits.reshape(model_input_shape)

    return digits


def get_mnist_digits_data_TF(num_training=59000, num_validation=1000, num_test=10000):
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_digits_data_inner(num_training, num_validation, num_test)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    x_train = np.divide(x_train, 255)
    x_test = np.divide(x_test, 255)
    x_val = np.divide(x_val, 255)

    # y_train = to_categorical(y_train, num_classes)
    # y_val = to_categorical(y_val, num_classes)
    # y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_emnist_letters_data_for_CNN(num_training=122000, num_validation=2800, num_test=20800):
    x_train, y_train = emnist.extract_training_samples('letters')
    x_test, y_test = emnist.extract_test_samples('letters')
    y_train = np.subtract(y_train, 1)
    y_test = np.subtract(y_test, 1)

    # Выборка данных
    mask = list(range(num_training, num_training + num_validation))
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    x_test = x_test[mask]
    y_test = y_test[mask]

    x_train = np.divide(x_train, 255).astype("float64")
    x_val = np.divide(x_val, 255).astype("float64")
    x_test = np.divide(x_test, 255).astype("float64")

    x_train = x_train[:, np.newaxis, :, :]
    x_val = x_val[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_val,
        'y_val': y_val,
        'X_test': x_test,
        'y_test': y_test
    }

    return data


def get_emnist_letters_data(num_training=122000, num_validation=2800, num_test=20800):
    x_train, y_train = emnist.extract_training_samples('letters')
    x_test, y_test = emnist.extract_test_samples('letters')
    y_train = np.subtract(y_train, 1)
    y_test = np.subtract(y_test, 1)

    # Выборка данных
    mask = list(range(num_training, num_training + num_validation))
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    x_test = x_test[mask]
    y_test = y_test[mask]

    # Вытягивание матриц в векторы
    x_train = x_train.reshape(num_training, -1).astype("float32")
    x_val = x_val.reshape(num_validation, -1).astype("float32")
    x_test = x_test.reshape(num_test, -1).astype("float32")

    x_train = x_train / 255
    x_test = x_test / 255
    x_val = x_val / 255

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_val,
        'y_val': y_val,
        'X_test': x_test,
        'y_test': y_test
    }

    return data


def get_mnist_digits_data_inner(num_training=59000, num_validation=1000, num_test=10000):
    x_train = mnist.train_images()
    y_train = mnist.train_labels()
    x_test = mnist.test_images()
    y_test = mnist.test_labels()

    # Выборка данных
    mask = list(range(num_training, num_training + num_validation))
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    x_test = x_test[mask]
    y_test = y_test[mask]

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_mnist_digits_data(num_training=59000, num_validation=1000, num_test=10000):
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_digits_data_inner(num_training, num_validation, num_test)

    # Вытягивание матриц в векторы
    x_train = x_train.reshape(num_training, -1).astype("float32")
    x_val = x_val.reshape(num_validation, -1).astype("float32")
    x_test = x_test.reshape(num_test, -1).astype("float32")

    x_train = x_train / 255
    x_test = x_test / 255
    x_val = x_val / 255

    data = {
        'X_train': x_train,
        'y_train': y_train,
        'X_val': x_val,
        'y_val': y_val,
        'X_test': x_test,
        'y_test': y_test
    }

    return data


def get_nist_sd19_digit_data_amount():
    path_to_dataset = config.NIST_SD19_ISOLATED_DIGITS_SRC
    train_data_len = 0
    test_data_len = 0
    val_data_len = 0
    total_test_and_val = 0
    for root_folder in os.listdir(path_to_dataset):
        if os.path.isdir(os.path.join(path_to_dataset, root_folder)):
            for folder in config.NIST_SD19_ISOLATED_DIGITS_TRAIN_FOLDERS.split(","):
                train_mit_file = os.path.join(path_to_dataset, root_folder, folder + ".mit")
                if os.path.isfile(train_mit_file):
                    f = open(train_mit_file, "r")
                    amount = f.readline()
                    train_data_len += int(amount)

            test_and_val_mit_file = os.path.join(path_to_dataset, root_folder,
                                                 config.NIST_SD19_ISOLATED_DIGITS_VAL_AND_TEST_FOLDER + ".mit")
            if os.path.isfile(test_and_val_mit_file):
                f = open(test_and_val_mit_file, "r")
                total_amount = int(f.readline())
                total_test_and_val += total_amount

                current_test_data_len = round(total_amount / 2)
                current_val_data_len = total_amount - current_test_data_len
                test_data_len += current_test_data_len
                val_data_len += current_val_data_len

    return train_data_len, val_data_len, test_data_len


def save_single_sd19_digits_data():
    train_index = 0
    test_index = 0
    val_index = 0
    path_to_dataset = config.NIST_SD19_ISOLATED_DIGITS_SRC
    train_data_len, val_data_len, test_data_len = get_nist_sd19_digit_data_amount()
    with h5py.File("single_dgts_data_64_UINT8.h5", "w") as h5f:

        train_shape = (train_data_len, 64, 64)
        test_shape = (test_data_len, 64, 64)
        val_shape = (val_data_len, 64, 64)

        h5f.create_dataset("train", shape=train_shape, dtype=np.uint8, compression="gzip")
        h5f.create_dataset("train_labels", shape=(train_data_len,), dtype=np.uint8, compression="gzip")

        h5f.create_dataset("test", shape=test_shape, dtype=np.uint8, compression="gzip")
        h5f.create_dataset("test_labels", shape=(test_data_len,), dtype=np.uint8, compression="gzip")

        h5f.create_dataset("val", shape=val_shape, dtype=np.uint8, compression="gzip")
        h5f.create_dataset("val_labels", shape=(val_data_len,), dtype=np.uint8, compression="gzip")

    for root_folder in os.listdir(path_to_dataset):
        current_label = int(codecs.decode(root_folder, "hex"))
        if os.path.isdir(os.path.join(path_to_dataset, root_folder)):
            for folder in config.NIST_SD19_ISOLATED_DIGITS_TRAIN_FOLDERS.split(","):
                train_folder = os.path.join(path_to_dataset, root_folder, folder)
                if os.path.isdir(train_folder):
                    train_batch_data = read_batch_nist_sd19_data(train_folder)
                    train_batch_len = train_batch_data.shape[0]

                    with h5py.File("single_dgts_data_64_UINT8.h5", "a") as f:
                        train_data_64 = f["train"]
                        train_labels_64 = f["train_labels"]
                        train_data_64[train_index: train_index + train_batch_len, :] = train_batch_data
                        train_labels_64[train_index:train_index + train_batch_len] = current_label

                    train_index += train_batch_len

                    train_batch_data = None
                    del train_batch_data

            val_and_test_folder = os.path.join(path_to_dataset, root_folder,
                                               config.NIST_SD19_ISOLATED_DIGITS_VAL_AND_TEST_FOLDER)

            val_and_test_batch = read_batch_nist_sd19_data(val_and_test_folder)
            val_and_test_batch_len = val_and_test_batch.shape[0]
            test_batch_len = round(val_and_test_batch_len / 2)
            val_batch_len = val_and_test_batch_len - test_batch_len

            with h5py.File("single_dgts_data_64_UINT8.h5", "a") as f:
                test_data_64 = f["test"]
                test_labels_64 = f["test_labels"]
                test_data_64[test_index: test_index + test_batch_len] = val_and_test_batch[:test_batch_len]
                test_labels_64[test_index: test_index + test_batch_len] = current_label

                test_index += test_batch_len

                val_data_64 = f["val"]
                val_labels_64 = f["val_labels"]
                val_data_64[val_index: val_index + val_batch_len] = \
                    val_and_test_batch[test_batch_len:test_batch_len + val_batch_len]

                val_labels_64[val_index: val_index + val_batch_len] = current_label
                val_index += val_batch_len

            val_and_test_batch = None
            del val_and_test_batch


def read_batch_nist_sd19_data(path_to_datafolder):
    list_dir = os.listdir(path_to_datafolder)
    batch_len = len(list_dir)
    batch_data = np.zeros((batch_len, 64, 64))

    for idx, img_filename in enumerate(list_dir):
        img_path = os.path.join(path_to_datafolder, img_filename)
        digit_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        binary_inv = cv2.bitwise_not(digit_img)
        binary_64 = cv2.resize(binary_inv, (64, 64), interpolation=cv2.INTER_AREA)
        batch_data[idx, :] = binary_64

    return batch_data


def read_batch_touching_dgts(metadata_batch, labels_dict, root_data_path):
    batch_size = len(metadata_batch)
    batch_data = np.zeros((batch_size, 64, 64))
    batch_labels = np.zeros(batch_size, dtype=np.uint8)
    batch_idx = 0

    for _, metadata in metadata_batch.iterrows():
        full_path = os.path.join(root_data_path, metadata["filepath"])
        digits_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        binary_inv = cv2.bitwise_not(digits_img)
        binary_64 = cv2.resize(binary_inv, (64, 64), interpolation=cv2.INTER_AREA)

        batch_data[batch_idx, :] = binary_64
        string_label = metadata["label"]
        batch_labels[batch_idx] = labels_dict[string_label]
        batch_idx += 1

    return batch_data, batch_labels


def save_2touching_nist_digits():
    result_filename = "2touching_dgts_data_64_UINT8_last.h5"
    root_data_path = config.NIST_TOUCHING_DIGITS_SRC
    train_metadata = pd.read_csv(config.NIST_TOUCHING_DIGITS_2D_TRAIN, sep=" ", names=["filepath", "label"],
                                 dtype=np.str)

    test_metadata = pd.read_csv(config.NIST_TOUCHING_DIGITS_2D_TEST, sep=" ", names=["filepath", "label"], dtype=np.str)
    val_metadata = pd.read_csv(config.NIST_TOUCHING_DIGITS_2D_VAL, sep=" ", names=["filepath", "label"], dtype=np.str)

    train_data_len = len(train_metadata)
    test_data_len = len(test_metadata)
    val_data_len = len(val_metadata)

    with h5py.File(result_filename, "w") as h5f:
        train_shape = (train_data_len, 64, 64)
        test_shape = (test_data_len, 64, 64)
        val_shape = (val_data_len, 64, 64)

        h5f.create_dataset("train", shape=train_shape, dtype=np.uint8, compression="gzip", chunks=True)
        h5f.create_dataset("train_labels", shape=(train_data_len,), dtype=np.uint8, compression="gzip")

        h5f.create_dataset("test", shape=test_shape, dtype=np.uint8, compression="gzip", chunks=True)
        h5f.create_dataset("test_labels", shape=(test_data_len,), dtype=np.uint8, compression="gzip")

        h5f.create_dataset("val", shape=val_shape, dtype=np.uint8, compression="gzip", chunks=True)
        h5f.create_dataset("val_labels", shape=(val_data_len,), dtype=np.uint8, compression="gzip")

    labels = train_metadata["label"].unique()
    encoded_labels = LabelEncoder().fit_transform(labels)
    labels_dict = dict(zip(labels, encoded_labels))
    inv_labels_dict = dict(zip(encoded_labels, labels))

    # save train data
    batch_size = 20000
    dataset_names = ("train", "train_labels")

    save_touching_dgts_by_batches(batch_size, root_data_path, train_metadata, labels_dict, result_filename,
                                  dataset_names)

    # save test data
    dataset_names = ("test", "test_labels")
    batch_size = 10000
    save_touching_dgts_by_batches(batch_size, root_data_path, test_metadata, labels_dict, result_filename,
                                  dataset_names)

    # save val data
    dataset_names = ("val", "val_labels")
    batch_size = 10000
    save_touching_dgts_by_batches(batch_size, root_data_path, val_metadata, labels_dict, result_filename,
                                  dataset_names)


def save_3touching_nist_digits():
    root_data_path = config.NIST_TOUCHING_DIGITS_SRC

    train_metadata = pd.read_csv(config.NIST_TOUCHING_DIGITS_3D_TRAIN, sep=" ", names=["filepath", "label"],
                                 dtype=np.str)
    train_metadata = train_metadata[train_metadata["label"].str.startswith('10')]

    test_metadata = pd.read_csv(config.NIST_TOUCHING_DIGITS_3D_TEST, sep=" ", names=["filepath", "label"], dtype=np.str)
    test_metadata = test_metadata[test_metadata["label"].str.startswith('10')]

    val_metadata = pd.read_csv(config.NIST_TOUCHING_DIGITS_3D_VAL, sep=" ", names=["filepath", "label"], dtype=np.str)
    val_metadata = val_metadata[val_metadata["label"].str.startswith('10')]

    train_data_len = len(train_metadata)
    test_data_len = len(test_metadata)
    val_data_len = len(val_metadata)

    result_filename = "3touching_dgts_data_64_UINT8_1.h5"

    with h5py.File(result_filename, "w") as h5f:
        train_shape = (train_data_len, 64, 64)
        test_shape = (test_data_len, 64, 64)
        val_shape = (val_data_len, 64, 64)

        h5f.create_dataset("train", shape=train_shape, dtype=np.uint8, compression="gzip", chunks=True,
                           compression_opts=1)
        h5f.create_dataset("train_labels", shape=(train_data_len,), dtype=np.uint8, compression="gzip",
                           compression_opts=1)

        h5f.create_dataset("test", shape=test_shape, dtype=np.uint8, compression="gzip", chunks=True,
                           compression_opts=1)
        h5f.create_dataset("test_labels", shape=(test_data_len,), dtype=np.uint8, compression="gzip",
                           compression_opts=1)

        h5f.create_dataset("val", shape=val_shape, dtype=np.uint8, compression="gzip", chunks=True, compression_opts=1)
        h5f.create_dataset("val_labels", shape=(val_data_len,), dtype=np.uint8, compression="gzip", compression_opts=1)

    labels = train_metadata["label"].unique()
    encoded_labels = LabelEncoder().fit_transform(labels)
    labels_dict = dict(zip(labels, encoded_labels))
    inv_labels_dict = dict(zip(encoded_labels, labels))

    # save train data
    batch_size = 1000
    dataset_names = ("train", "train_labels")

    save_touching_dgts_by_batches(batch_size, root_data_path, train_metadata, labels_dict, result_filename,
                                  dataset_names)

    # save test data
    dataset_names = ("test", "test_labels")
    batch_size = 1000
    save_touching_dgts_by_batches(batch_size, root_data_path, test_metadata, labels_dict, result_filename,
                                  dataset_names)

    # save val data
    dataset_names = ("val", "val_labels")
    batch_size = 1000
    save_touching_dgts_by_batches(batch_size, root_data_path, val_metadata, labels_dict, result_filename,
                                  dataset_names)


def save_touching_dgts_by_batches(batch_size, root_data_path, metadata, labels_dict, result_filename, ds_names):
    data_len = len(metadata)
    batch_amount = int(data_len / batch_size) + 1
    batch_start_indexes = [i * batch_size for i in range(batch_amount)]
    batch_end_indexes = [i * batch_size + batch_size for i in range(batch_amount)]
    batch_end_indexes[-1] = data_len

    for start_indx, end_indx in zip(batch_start_indexes, batch_end_indexes):
        metadata_batch = metadata[start_indx:end_indx]
        batch_data, batch_labels = read_batch_touching_dgts(metadata_batch, labels_dict, root_data_path)

        with h5py.File(result_filename, "a") as f:
            f[ds_names[0]][start_indx:end_indx, :] = batch_data
            f[ds_names[1]][start_indx:end_indx] = batch_labels

            k = 0

        print("Saved {0:0.2f}%".format((end_indx / data_len) * 100))
        batch_data, batch_labels = None, None
        del batch_data, batch_labels


def get_hdf5_datasets_length(filename, datasets=('train', 'test', 'val')):
    with h5py.File(filename, "r") as f:
        train_len = len(f[datasets[0]])
        test_len = len(f[datasets[1]])
        val_len = len(f[datasets[2]])

    return train_len, test_len, val_len


def save_collected_touchingdgts_reduced_for_test():
    result_filename = "TouchingDgts_Dataset_test.h5"
    train = None
    train_labels = None
    val = None
    val_labels = None

    with h5py.File('3touching_dgts_data_64_UINT8.h5', 'r') as f:
        train = f['train'][:]
        train = np.divide(train, 255).astype("float32")
        train = train.reshape(train.shape[0], train.shape[1], train.shape[2], 1)
        train_labels = f['train_labels'][:]

        val = f['val'][:]
        val = np.divide(val, 255).astype("float32")
        val = val.reshape(val.shape[0], val.shape[1], val.shape[2], 1)
        val_labels = f['val_labels'][:]

    train_shape = (len(train), 64, 64, 1)
    train_labels_shape = (len(train_labels),)
    val_shape = (len(val), 64, 64, 1)
    val_labels_shape = (len(val_labels),)

    with h5py.File(result_filename, 'w') as f:
        f.create_dataset("train", shape=train_shape, dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=4, data=train)
        f.create_dataset("train_labels", shape=train_labels_shape, dtype=np.uint8, compression="gzip",
                         compression_opts=4, data=train_labels)

        f.create_dataset("val", shape=val_shape, dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=4, data=val)
        f.create_dataset("val_labels", shape=val_labels_shape, dtype=np.uint8, compression="gzip",
                         compression_opts=4, data=val_labels)


def save_mnist_data_for_test():
    result_filename = "mnist_test_data.h5"

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_digits_data_inner()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    x_train = np.divide(x_train, 255).astype("float32")
    x_test = np.divide(x_test, 255).astype("float32")
    x_val = np.divide(x_val, 255).astype("float32")

    with h5py.File(result_filename, 'w') as f:
        f.create_dataset("train", dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=4, data=x_train)
        f.create_dataset("train_labels", dtype=np.uint8, compression="gzip",
                         compression_opts=4, data=y_train)

        f.create_dataset("val", dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=4, data=x_val)
        f.create_dataset("val_labels", dtype=np.uint8, compression="gzip",
                         compression_opts=4, data=y_val)

        f.create_dataset("test", dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=4, data=x_test)
        f.create_dataset("test_labels", dtype=np.uint8, compression="gzip",
                         compression_opts=4, data=y_test)


def reduce_dataset_length(train_len, test_len, val_len):
    train_len = int(train_len / 5)
    test_len = int(test_len / 5)
    val_len = int(val_len / 5)

    return train_len, test_len, val_len


def read_batch_2touching_dgts(data_filename, ds_names, start_index, end_index):
    dataset_data_name = ds_names[0]
    dataset_label_name = ds_names[1]

    with h5py.File(data_filename, "r") as f:
        data = f[dataset_data_name][start_index:end_index]
        labels = f[dataset_label_name][start_index:end_index]

    # convert to float32 data

    data = np.divide(data, 255).astype("float32")
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    return data, labels


def save_2touching_dgts_by_batches(batch_size, data_len, data_filename, result_filename, ds_names):
    batch_amount = int(data_len / batch_size) + 1
    batch_start_indexes = [i * batch_size for i in range(batch_amount)]
    batch_end_indexes = [i * batch_size + batch_size for i in range(batch_amount)]
    batch_end_indexes[-1] = data_len

    for start_indx, end_indx in zip(batch_start_indexes, batch_end_indexes):
        batch_data, batch_labels = read_batch_2touching_dgts(data_filename, ds_names, start_indx, end_indx)

        with h5py.File(result_filename, "a") as f:
            f[ds_names[0]][start_indx:end_indx, :] = batch_data
            f[ds_names[1]][start_indx:end_indx] = batch_labels

        print("Saved {0:0.2f}%".format((end_indx / data_len) * 100))
        batch_data, batch_labels = None, None
        del batch_data, batch_labels


def save_2touching_dgts_data_64_in_float32():
    result_filename = "2touching_dgts_data_64_float32_zip2.h5"

    input_filename = "2touching_dgts_data_64_UINT8.h5"

    dgts_2touch_train_len, dgts_2touch_test_len, dgts_2touch_val_len = get_hdf5_datasets_length(input_filename)

    with h5py.File(result_filename, "w") as f:
        train_shape = (dgts_2touch_train_len, 64, 64, 1)
        test_shape = (dgts_2touch_test_len, 64, 64, 1)
        val_shape = (dgts_2touch_val_len, 64, 64, 1)

        f.create_dataset("train", shape=train_shape, dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=2)
        f.create_dataset("train_labels", shape=(dgts_2touch_train_len,), dtype=np.uint8, compression="gzip",
                         compression_opts=2)

        f.create_dataset("test", shape=test_shape, dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=2)
        f.create_dataset("test_labels", shape=(dgts_2touch_test_len,), dtype=np.uint8, compression="gzip",
                         compression_opts=2)

        f.create_dataset("val", shape=val_shape, dtype=np.float32, compression="gzip", chunks=True, compression_opts=2)
        f.create_dataset("val_labels", shape=(dgts_2touch_val_len,), dtype=np.uint8, compression="gzip",
                         compression_opts=2)

        batch_size = 30000
        dataset_names = ('train', 'train_labels')
        save_2touching_dgts_by_batches(batch_size, dgts_2touch_train_len, input_filename, result_filename,
                                       dataset_names)

        batch_size = 10000
        dataset_names = ('test', 'test_labels')
        save_2touching_dgts_by_batches(batch_size, dgts_2touch_test_len, input_filename, result_filename,
                                       dataset_names)
        batch_size = 10000
        dataset_names = ('val', 'val_labels')
        save_2touching_dgts_by_batches(batch_size, dgts_2touch_val_len, input_filename, result_filename,
                                       dataset_names)


def collect_touching_digits():
    result_filename = "TouchingDgts_Dataset_64_f32_reduced_x5_zip4.h5"

    dgts_single_filename = "single_dgts_data_64_UINT8.h5"
    dgts_2touch_filename = "2touching_dgts_data_64_UINT8.h5"
    dgts_3touch_filename = "3touching_dgts_data_64_UINT8.h5"
    # read single digits
    single_train_len, single_test_len, single_val_len = reduce_dataset_length(
        *get_hdf5_datasets_length(dgts_single_filename))

    # read 2touching digits
    dgts_2touch_train_len, dgts_2touch_test_len, dgts_2touch_val_len = reduce_dataset_length(
        *get_hdf5_datasets_length(dgts_2touch_filename))

    # read 3touching digits
    dgts_3touch_train_len, dgts_3touch_test_len, dgts_3touch_val_len = reduce_dataset_length(
        *get_hdf5_datasets_length(dgts_3touch_filename))

    # collect to dataset "TouchingDGTS_LClsfr"
    result_train_len = single_train_len + dgts_2touch_train_len + dgts_3touch_train_len
    result_test_len = single_test_len + dgts_2touch_test_len + dgts_3touch_test_len
    result_val_len = single_val_len + dgts_2touch_val_len + dgts_3touch_val_len

    with h5py.File(result_filename, "w") as f:
        train_shape = (result_train_len, 64, 64, 1)
        test_shape = (result_test_len, 64, 64, 1)
        val_shape = (result_val_len, 64, 64, 1)

        f.create_dataset("train", shape=train_shape, dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=4)
        f.create_dataset("train_labels", shape=(result_train_len,), dtype=np.uint8, compression="gzip",
                         compression_opts=4)

        f.create_dataset("test", shape=test_shape, dtype=np.float32, compression="gzip", chunks=True,
                         compression_opts=4)
        f.create_dataset("test_labels", shape=(result_test_len,), dtype=np.uint8, compression="gzip",
                         compression_opts=4)

        f.create_dataset("val", shape=val_shape, dtype=np.float32, compression="gzip", chunks=True, compression_opts=4)
        f.create_dataset("val_labels", shape=(result_val_len,), dtype=np.uint8, compression="gzip", compression_opts=4)

    batch_size = 1000
    save_collected_data(batch_size, ('train', 'train_labels'),
                        dgts_single_filename, dgts_2touch_filename, dgts_3touch_filename,
                        single_train_len, dgts_2touch_train_len, dgts_3touch_train_len,
                        result_filename, result_train_len)

    save_collected_data(batch_size, ('val', 'val_labels'),
                        dgts_single_filename, dgts_2touch_filename, dgts_3touch_filename,
                        single_val_len, dgts_2touch_val_len, dgts_3touch_val_len,
                        result_filename, result_val_len)

    save_collected_data(batch_size, ('test', 'test_labels'),
                        dgts_single_filename, dgts_2touch_filename, dgts_3touch_filename,
                        single_test_len, dgts_2touch_test_len, dgts_3touch_test_len,
                        result_filename, result_test_len)

    kek = 0

    return


def save_collected_data(batch_size, ds_names, dgts_single_filename, dgts_2touch_filename, dgts_3touch_filename,
                        single_train_len, dgts_2touch_train_len, dgts_3touch_train_len,
                        result_filename, result_train_len):
    batch_amount_single = int(single_train_len / batch_size) + 1
    batch_start_indexes_single = [i * batch_size for i in range(batch_amount_single)]
    batch_end_indexes_single = [i * batch_size + batch_size for i in range(batch_amount_single)]
    batch_end_indexes_single[-1] = single_train_len

    save_collected_data_internal(ds_names, batch_amount_single,
                                 batch_start_indexes_single, batch_end_indexes_single,
                                 batch_start_indexes_single, batch_end_indexes_single,
                                 dgts_single_filename, result_filename, result_train_len, label_value=0)

    batch_amount_2touching = int(dgts_2touch_train_len / batch_size) + 1
    batch_start_indexes_2touching = [i * batch_size for i in range(batch_amount_2touching)]
    batch_end_indexes_2touching = [i * batch_size + batch_size for i in range(batch_amount_2touching)]
    batch_end_indexes_2touching[-1] = dgts_2touch_train_len
    batch_start_indexes_result = np.add(batch_start_indexes_2touching, batch_end_indexes_single[-1])
    batch_end_indexes_result = np.add(batch_end_indexes_2touching, batch_end_indexes_single[-1])

    save_collected_data_internal(ds_names, batch_amount_2touching,
                                 batch_start_indexes_2touching, batch_end_indexes_2touching,
                                 batch_start_indexes_result, batch_end_indexes_result,
                                 dgts_2touch_filename, result_filename, result_train_len, label_value=1)

    batch_amount_3touching = int(dgts_3touch_train_len / batch_size) + 1
    batch_start_indexes_3touching = [i * batch_size for i in range(batch_amount_3touching)]
    batch_end_indexes_3touching = [i * batch_size + batch_size for i in range(batch_amount_3touching)]
    batch_end_indexes_3touching[-1] = dgts_3touch_train_len
    batch_start_indexes_result = np.add(batch_start_indexes_3touching, batch_end_indexes_result[-1])
    batch_end_indexes_result = np.add(batch_end_indexes_3touching, batch_end_indexes_result[-1])

    save_collected_data_internal(ds_names, batch_amount_3touching,
                                 batch_start_indexes_3touching, batch_end_indexes_3touching,
                                 batch_start_indexes_result, batch_end_indexes_result,
                                 dgts_3touch_filename, result_filename, result_train_len, label_value=2)


def save_collected_data_internal(ds_names, batch_amount,
                                 start_indexes, end_indexes,
                                 start_indexes_res, end_indexes_res,
                                 data_filename, result_filename, result_val_len, label_value):
    dataset_data_name = ds_names[0]
    dataset_labels_name = ds_names[1]

    for idx in range(batch_amount):
        data = None
        labels = None

        with h5py.File(data_filename, "r") as f:
            data = f[dataset_data_name][start_indexes[idx]:end_indexes[idx]]
            labels = np.zeros(data.shape[0])
            labels[:] = label_value  # single digits label

        # convert to float32 data

        data = np.divide(data, 255).astype("float32")
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

        with h5py.File(result_filename, "a") as res:
            res[dataset_data_name][start_indexes_res[idx]:end_indexes_res[idx], :] = data
            res[dataset_labels_name][start_indexes_res[idx]:end_indexes_res[idx]] = labels

        print("Saved {0:0.2f}%".format((end_indexes_res[idx] / result_val_len) * 100))
        data, labels = None, None
        del data, labels


def get_random_indexes_of_digits(labels):
    unique_labels = set(labels)
    random_indexes = {}
    for label in unique_labels:
        label_idxs = np.where(labels == label)[0]
        rand_idx = np.random.randint(0, label_idxs.shape[0])
        new_idx = label_idxs[rand_idx]

        random_indexes.update({label: new_idx})

    return random_indexes


def get_random_indexes_of_data(labels, label=0, size=10):
    label_idxs = np.where(labels == label)[0]
    rand_idxs = np.random.randint(0, label_idxs.shape[0], size=size)
    new_idxs = label_idxs[rand_idxs]

    return new_idxs


# 30044 - 0, 83951 -1 , 30045 - 1 , 83952 - 2, 123129 - 2

# for train : 384686 - 1, 502671 - 2
# for test: 85134 - 1, 125208 - 2
# for val: 83951 -1, 123129 - 2

# with h5py.File("2touching_dgts_data_64_float32_zip2.h5", "a") as f:
#     train = f["train"]
#     test = f["test"]
#     val = f["val"]
#
#     train_labels = f["train_labels"][:]
#     test_labels = f["test_labels"][:]
#     val_labels = f["val_labels"][:]
#
#     random_train_labels_idxs = get_random_indexes_of_digits(train_labels)
#     random_test_labels_idxs = get_random_indexes_of_digits(test_labels)
#     random_val_labels_idxs = get_random_indexes_of_digits(val_labels)
#
#     last_train_label = 75
#     last_train = train[random_train_labels_idxs[last_train_label]]
#
#     last_val_label = 88
#     last_val = val[random_val_labels_idxs[last_val_label]]
#
#     last_test_label = 67
#     last_test = test[random_test_labels_idxs[last_test_label]]
#
#     plt.subplot(2, 2, 1)
#     plt.title('train')
#     plt.imshow(last_train.reshape((64, 64)), cmap='gray')
#     plt.subplot(2, 2, 2)
#     plt.title('val')
#     plt.imshow(last_val.reshape((64, 64)), cmap='gray')
#     plt.subplot(2, 2, 3)
#     plt.title('test')
#     plt.imshow(last_test.reshape((64, 64)), cmap='gray')
#     plt.show()
#
#     train[-1, :] = last_train
#     f["train_labels"][-1] = last_train_label
#
#     val[-1, :] = last_val
#     f["val_labels"][-1] = last_val_label
#
#     test[-1, :] = last_test
#     f["test_labels"][-1] = last_test_label

# with h5py.File("2touching_dgts_data_64_float32_zip2.h5", "r") as f:
#     train = f["train"]
#     test = f["test"]
#     val = f["val"]
#
#     train_labels = f["train_labels"][:]
#     test_labels = f["test_labels"][:]
#     val_labels = f["val_labels"][:]
#
#     # random_train_labels_idxs = get_random_indexes_of_digits(train_labels)
#     # random_test_labels_idxs = get_random_indexes_of_digits(test_labels)
#     # random_val_labels_idxs = get_random_indexes_of_digits(val_labels)
#
#     random_train_labels_idxs = get_random_indexes_of_data(train_labels, label=85)
#     random_test_labels_idxs = get_random_indexes_of_data(test_labels, label=96)
#     random_val_labels_idxs = get_random_indexes_of_data(val_labels, label=79)
#
#     # train = train[0:len(train) - 1]
#     # print(len(test))
#     # print(len(val))
#     # last_val = val[-1]
#     # last_train = train[-1]
#     # last_test = test[-1]
#     # print(len(train))
#     # plt.subplot(2, 2, 1)
#     # plt.title('train')
#     # plt.imshow(last_train.reshape((64, 64)), cmap='gray')
#     # plt.subplot(2, 2, 2)
#     # plt.title('val')
#     # plt.imshow(last_val.reshape((64, 64)), cmap='gray')
#     # plt.subplot(2, 2, 3)
#     # plt.title('test')
#     # plt.imshow(last_test.reshape((64, 64)), cmap='gray')
#     # plt.show()
#     for idx in range(0, len(random_train_labels_idxs)):
#         plt.subplot(2, 2, 1)
#         plt.title('train')
#         plt.imshow(train[random_train_labels_idxs[idx]].reshape((64, 64)), cmap='gray')
#         plt.subplot(2, 2, 2)
#         plt.title('val')
#         plt.imshow(val[random_val_labels_idxs[idx]].reshape((64, 64)), cmap='gray')
#         plt.subplot(2, 2, 3)
#         plt.title('test')
#         plt.imshow(test[random_test_labels_idxs[idx]].reshape((64, 64)), cmap='gray')
#         plt.show()

#
# for idx in random_test_labels_idxs:
#     plt.imshow(test[idx], cmap='gray')
#     plt.show()
#
# for idx in random_val_labels_idxs:
#     plt.imshow(val[idx], cmap='gray')
#     plt.show()
# save_2touching_nist_digits()
# save_3touching_nist_digits()
# save_collected_touchingdgts_reduced_for_test()


# with h5py.File('TouchingDgts_Dataset_test.h5', 'r') as f:
#     train = f['train'][:]
#     train_labels = f['train_labels'][:]
#     print(len(train))
# save_collected_touchingdgts_reduced_for_test()
# collect_touching_digits()
# save_2touching_dgts_data_64_in_float32()
