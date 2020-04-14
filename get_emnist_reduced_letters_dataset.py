import numpy as np
import emnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import cv2


def emnsit_uppercase_label_filter(a):
    return ((a >= 10) & (a < 16) | (a == 33))


def label_filter(a):
    return ((a >= 0) & (a < 6) | (a == 23))


def replace_x_letter_label(a):
    for idx, el in enumerate(a):
        if el == 23:
            a[idx] = 6

    return a


def save_emnist_uppercase_reduced_letters64_dataset():
    x_train, y_train = emnist.extract_training_samples('byclass')
    x_test, y_test = emnist.extract_test_samples('byclass')

    train_mask = emnsit_uppercase_label_filter(y_train)
    test_mask = emnsit_uppercase_label_filter(y_test)

    x_train_reduced = x_train[train_mask]
    x_train_reduced = [cv2.resize(i, (64, 64), interpolation=cv2.INTER_NEAREST) for i in x_train_reduced]
    y_train_reduced = y_train[train_mask]
    # shift to 0 label
    y_train_reduced -= 10
    y_train_reduced = replace_x_letter_label(y_train_reduced)

    x_test_reduced = x_test[test_mask]
    x_test_reduced = [cv2.resize(i, (64, 64), interpolation=cv2.INTER_NEAREST) for i in x_test_reduced]
    y_test_reduced = y_test[test_mask]
    y_test_reduced -= 10
    y_test_reduced = replace_x_letter_label(y_test_reduced)

    x_train_reduced, x_val_reduced, y_train_reduced, y_val_reduced = train_test_split(x_train_reduced, y_train_reduced,
                                                                                      test_size=0.1)
    x_train_reduced = np.divide(x_train_reduced, 255).astype("float64")
    x_val_reduced = np.divide(x_val_reduced, 255).astype("float64")
    x_test_reduced = np.divide(x_test_reduced, 255).astype("float64")
    #
    x_train_reduced = x_train_reduced.reshape(x_train_reduced.shape[0], x_train_reduced.shape[1],
                                              x_train_reduced.shape[2],
                                              1)

    x_val_reduced = x_val_reduced.reshape(x_val_reduced.shape[0], x_val_reduced.shape[1], x_val_reduced.shape[2], 1)
    x_test_reduced = x_test_reduced.reshape(x_test_reduced.shape[0], x_test_reduced.shape[1], x_test_reduced.shape[2],
                                            1)

    letters_dataset = {
        "x_train": x_train_reduced,
        "y_train": y_train_reduced,
        "x_val": x_val_reduced,
        "y_val": y_val_reduced,
        "x_test": x_test_reduced,
        "y_test": y_test_reduced
    }

    with open("eng_uppercase_letters64_dataset.bin", "wb") as file:
        pickle.dump(letters_dataset, file)


def save_emnist_uppercase_reduced_letters32_dataset():
    x_train, y_train = emnist.extract_training_samples('byclass')
    x_test, y_test = emnist.extract_test_samples('byclass')

    train_mask = emnsit_uppercase_label_filter(y_train)
    test_mask = emnsit_uppercase_label_filter(y_test)

    x_train_reduced = x_train[train_mask]
    y_train_reduced = y_train[train_mask]
    # shift to 0 label
    y_train_reduced -= 10
    y_train_reduced = replace_x_letter_label(y_train_reduced)

    x_test_reduced = x_test[test_mask]
    y_test_reduced = y_test[test_mask]
    y_test_reduced -= 10
    y_test_reduced = replace_x_letter_label(y_test_reduced)

    x_train_reduced, x_val_reduced, y_train_reduced, y_val_reduced = train_test_split(x_train_reduced, y_train_reduced,
                                                                                      test_size=0.1)
    x_train_reduced = np.divide(x_train_reduced, 255).astype("float64")
    x_val_reduced = np.divide(x_val_reduced, 255).astype("float64")
    x_test_reduced = np.divide(x_test_reduced, 255).astype("float64")
    #
    x_train_reduced = x_train_reduced.reshape(x_train_reduced.shape[0], x_train_reduced.shape[1],
                                              x_train_reduced.shape[2],
                                              1)

    x_val_reduced = x_val_reduced.reshape(x_val_reduced.shape[0], x_val_reduced.shape[1], x_val_reduced.shape[2], 1)
    x_test_reduced = x_test_reduced.reshape(x_test_reduced.shape[0], x_test_reduced.shape[1], x_test_reduced.shape[2],
                                            1)

    letters_dataset = {
        "x_train": x_train_reduced,
        "y_train": y_train_reduced,
        "x_val": x_val_reduced,
        "y_val": y_val_reduced,
        "x_test": x_test_reduced,
        "y_test": y_test_reduced
    }

    with open("eng_uppercase_letters32_dataset.bin", "wb") as file:
        pickle.dump(letters_dataset, file)


def save_emnist_reduced_letters_dataset():
    x_train, y_train = emnist.extract_training_samples('letters')
    x_test, y_test = emnist.extract_test_samples('letters')
    # Переход к меткам диапазона [0..25]
    y_train = np.subtract(y_train, 1)
    y_test = np.subtract(y_test, 1)

    # train_mask = label_filter(y_train)
    train_mask = label_filter(y_train)
    test_mask = label_filter(y_test)

    x_train_reduced = x_train[train_mask]
    y_train_reduced = y_train[train_mask]
    y_train_reduced = replace_x_letter_label(y_train_reduced)

    x_test_reduced = x_test[test_mask]
    y_test_reduced = y_test[test_mask]
    y_test_reduced = replace_x_letter_label(y_test_reduced)

    x_train_reduced, x_val_reduced, y_train_reduced, y_val_reduced = train_test_split(x_train_reduced, y_train_reduced,
                                                                                      test_size=0.1)

    x_train_reduced = np.divide(x_train_reduced, 255).astype("float64")
    x_val_reduced = np.divide(x_val_reduced, 255).astype("float64")
    x_test_reduced = np.divide(x_test_reduced, 255).astype("float64")
    #
    x_train_reduced = x_train_reduced.reshape(x_train_reduced.shape[0], x_train_reduced.shape[1],
                                              x_train_reduced.shape[2],
                                              1)
    x_val_reduced = x_val_reduced.reshape(x_val_reduced.shape[0], x_val_reduced.shape[1], x_val_reduced.shape[2], 1)
    x_test_reduced = x_test_reduced.reshape(x_test_reduced.shape[0], x_test_reduced.shape[1], x_test_reduced.shape[2],
                                            1)

    letters_dataset = {
        "x_train": x_train_reduced,
        "y_train": y_train_reduced,
        "x_val": x_val_reduced,
        "y_val": y_val_reduced,
        "x_test": x_test_reduced,
        "y_test": y_test_reduced
    }

    with open("eng_letters_dataset.bin", "wb") as file:
        pickle.dump(letters_dataset, file)


# a = np.where(y_train_reduced == 0)[0]
# b = np.where(y_train_reduced == 1)[0]
# c = np.where(y_train_reduced == 2)[0]
# d = np.where(y_train_reduced == 3)[0]
# e = np.where(y_train_reduced == 4)[0]
# f = np.where(y_train_reduced == 5)[0]
# x = np.where(y_train_reduced == 6)[0]

# for a_idx in a:
#     plt.imshow(x_train_reduced[a_idx])
#     plt.show()
# plt.subplot(4, 2, 1)
# plt.imshow(x_train_reduced[a[0]])
# plt.subplot(4, 2, 2)
# plt.imshow(x_train_reduced[b[0]])
# plt.subplot(4, 2, 3)
# plt.imshow(x_train_reduced[c[0]])
# plt.subplot(4, 2, 4)
# plt.imshow(x_train_reduced[d[0]])
# plt.subplot(4, 2, 5)
# plt.imshow(x_train_reduced[e[0]])
# plt.subplot(4, 2, 6)
# plt.imshow(x_train_reduced[f[0]])
# plt.subplot(4, 2, 7)
# plt.imshow(x_train_reduced[x[0]])
# plt.show()
# save_emnist_uppercase_reduced_letters64_dataset()


