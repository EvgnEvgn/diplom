import cv2
import numpy as np
from neural_nets.model_utils import *
from extracting_data import segment_date_from_col3
import os
import matplotlib.pyplot as plt


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

    # val_X = dataset["val_X"]
    # val_Y = dataset["val_Y"]

    test_X = dataset["test_X"]
    test_Y = dataset["test_Y"]

    return train_X, train_Y, test_X, test_Y


def show_tchng_dgts_dataset():
    train_X, train_Y, _, _ = get_tchng_dgts_small_dataset("neural_nets/dataset_lbl_0-9-00-31_val=test_64_float32.bin")
    test_X, test_Y = get_test_tchgn_dgts_dataset("test_date_numbers_50examples.bin")

    indxs = np.where((train_Y >= 0) & (train_Y <= 9))[0]

    np.random.shuffle(indxs)
    indxs_to_show = indxs[0:50]
    test_idx = 0
    for indx in indxs_to_show:
        plt.subplot(2, 1, 1)
        plt.imshow(train_X[indx].reshape(64, 64))
        plt.title("Label: " + str(train_Y[indx]))
        plt.subplot(2, 1, 2)
        plt.imshow(test_X[test_idx, :])
        plt.title("Label: " + str(test_Y[test_idx]))
        plt.show()
        test_idx += 1


def save_test_data_date_from_col3():
    dir_path = r"Output_temp/ExtractedColumns/3_col"
    img_paths = np.array(os.listdir(dir_path))
    indx = np.arange(len(img_paths))
    #np.random.seed(0)
    np.random.shuffle(indx)

    img_paths = img_paths[list(indx)]
    # Length_model = get_touching_dgts_L_classfier_tf_model()
    # Touching2Digit_model = get_touching_dgts_C2_classfier_tf_model()
    num_examples = 400
    test_X = np.zeros((num_examples, 64 * 64), dtype=np.float32)
    test_Y = np.zeros((num_examples), dtype=np.int64)
    idx = 0

    for img_path in img_paths:
        if idx >= num_examples:
            break

        result_path = os.path.join(dir_path, img_path)
        img = cv2.imread(result_path)
        segmented_imgs = segment_date_from_col3(img)

        if segmented_imgs is not None:
            for number_img in segmented_imgs:
                if idx >= num_examples:
                    break

                data = number_img

                plt.subplot(2, 2, 1)
                plt.imshow(img)
                plt.title('Orig')
                plt.subplot(2, 2, 2)
                plt.title('Segmented')
                plt.imshow(data)
                plt.show()

                res = input("Добавить в тест?322 - не добавляем, иначе вводим класс. stop - закончить: \n")
                if res == 'stop':
                    break
                res = int(res)

                if (res < 0 or res > 41) and res != 322:
                    print("ты даун, шо ты ввел! бб")
                    exit()
                if res != 322:
                    data = np.divide(data, 255).astype("float32")
                    data = np.reshape(data, (64 * 64))
                    test_X[idx, :] = data
                    test_Y[idx] = res
                    idx += 1
                    plt.close()
                    print("Количество добавленных: " + str(idx))
                else:
                    continue

    dataset = {"test_X": test_X, "test_Y": test_Y}
    with open("test_date_numbers_" + str(num_examples) + "examples_alt.bin", "wb") as file:
        pickle.dump(dataset, file)


save_test_data_date_from_col3()
#show_tchng_dgts_dataset()
