import mnist
# from neural_nets.features_utils import hog_feature, extract_features
import numpy as np
import emnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


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

# def get_digits_mnist_features_data(num_training=59000, num_validation=1000, num_test=10000):
#     x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_digits_data_inner(num_training, num_validation, num_test)
#
#     feature_fns = [lambda img: hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
#                                    visualise=False)]
#
#     x_train_feats = extract_features(x_train, feature_fns, verbose=True)
#     x_val_feats = extract_features(x_val, feature_fns)
#     x_test_feats = extract_features(x_test, feature_fns)
#
#     # list_hog_fd = []
#     # for feature in features:
#     #     fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
#     #              visualise=False)
#     #     list_hog_fd.append(fd)
#     # #hog_features = np.array(list_hog_fd, 'float64')
#
#     data = {
#         'X_train': x_train_feats,
#         'y_train': y_train,
#         'X_val': x_val_feats,
#         'y_val': y_val,
#         'X_test': x_test_feats,
#         'y_test': y_test
#     }
#
#     return data
