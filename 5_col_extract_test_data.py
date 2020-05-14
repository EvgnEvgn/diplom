import cv2
import numpy as np
from neural_nets.model_utils import *
from extracting_data import segment_touching_digits
import os
import matplotlib.pyplot as plt


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


def show_tchng_dgts_dataset():
    train_X, train_Y, _, _, _, _ = get_tchng_dgts_small_dataset("neural_nets/2tchg_dgts_10-99_64_val=test_float32.bin")

    indxs = np.where((train_Y >= 40) & (train_Y <= 49))[0]

    np.random.shuffle(indxs)
    indxs_to_show = indxs[0:50]
    for indx in indxs_to_show:
        plt.imshow(train_X[indx].reshape(64, 64))
        plt.title("Label: " + str(train_Y[indx]))
        plt.show()


def compare_tchng_dgts():
    dir_path = r"Output_temp/ExtractedColumns/5_col"
    img_paths = os.listdir(dir_path)
    # Length_model = get_touching_dgts_L_classfier_tf_model()
    # Touching2Digit_model = get_touching_dgts_C2_classfier_tf_model()
    num_examples = 50
    test_X = np.zeros((num_examples, 64, 64))
    test_Y = np.zeros((num_examples))
    idx = 0
    _, _, _, _, test_X, test_Y = get_tchng_dgts_small_dataset("neural_nets/touching_dgts_90-99_64.bin")

    for img_path in img_paths:
        if idx >= num_examples:
            break

        result_path = os.path.join(dir_path, img_path)
        img = cv2.imread(result_path)
        segmented_img = segment_touching_digits(img)
        if segmented_img is not None:
            data = np.reshape(segmented_img, (64, 64))
            # data = np.divide(data, 255).astype("float32")

            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.title('Orig')
            plt.subplot(2, 2, 2)
            plt.title('Segmented')
            plt.imshow(data)
            plt.subplot(2, 2, 3)
            plt.imshow(test_X[idx, :].reshape((64, 64)))
            plt.title('Tchg_Dgts: {}'.format(test_Y[idx]))
            plt.show()
            idx += 1


def extract_test_2tchg_dgts_data():
    dir_path = r"Output_temp/ExtractedColumns/5_col"
    img_paths = os.listdir(dir_path)
    # Length_model = get_touching_dgts_L_classfier_tf_model()
    # Touching2Digit_model = get_touching_dgts_C2_classfier_tf_model()
    num_examples = 500
    test_X = np.zeros((num_examples, 64, 64))
    test_Y = np.zeros((num_examples), dtype=np.int64)
    idx = 0
    for img_path in img_paths:

        if idx >= num_examples:
            break

        result_path = os.path.join(dir_path, img_path)
        img = cv2.imread(result_path)
        segmented_img = segment_touching_digits(img)
        if segmented_img is not None:
            data = np.reshape(segmented_img, (64, 64))
            # data = np.divide(data, 255).astype("float32")

            plt.subplot(2, 1, 1)
            plt.imshow(img)
            plt.title('Orig')
            plt.subplot(2, 1, 2)
            plt.imshow(data)
            plt.show()

            res = input("Добавить в тест?322 - не добавляем, иначе вводим класс. stop - закончить: \n")
            if res == 'stop':
                break
            res = int(res)

            if (res < 0 or res > 100) and res != 322:
                print("ты даун, шо ты ввел! бб")
                exit()
            if res != 322:
                data = np.divide(data, 255).astype("float32")
                test_X[idx:] = data
                test_Y[idx] = res
                idx += 1
                plt.close()
                print("Количество добавленных: " + str(idx))
            else:
                continue

    dataset = {"test_X": test_X, "test_Y": test_Y}
    with open("test_2tch_dgts_" + str(num_examples) + "examples.bin", "wb") as file:
        pickle.dump(dataset, file)


with open("test_2tch_dgts_500examples.bin", "rb") as file:
    dataset = pickle.load(file)

test_X = dataset["test_X"]
test_Y = dataset["test_Y"]

test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1] * test_X.shape[2]))
test_Y = np.subtract(test_Y, 10)
test_Y = np.array(test_Y, dtype=np.int64)

idxs = np.where(test_Y == 90)[0]

test_X = test_X[idxs]
test_Y = test_Y[idxs]
dataset = {"test_X": test_X, "test_Y": test_Y}

# with open("test_2tch_dgts_from_10-99_477examples_float32.bin", "wb") as file:
#     dataset = pickle.dump(dataset, file)

for i in range(len(test_Y)):
    plt.imshow(test_X[i].reshape(64, 64))
    plt.title("Label: " + str(test_Y[i]))
    plt.show()

# extract_test_2tchg_dgts_data()
#compare_tchng_dgts()
# show_tchng_dgts_dataset()
#extract_test_2tchg_dgts_data()
