import cv2
import numpy as np
from neural_nets.model_utils import *
from extracting_data import segment_rus_text_score_col6
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


def process_train_data_from_col6():
    try:
        with open("data_rus_text_score.bin", "rb") as file:
            dataset = pickle.load(file)
    except:
        print("нет данных:(")
        exit()
    extracted_data = dataset["data"]
    extracted_labels = dataset["labels"]
    train_w = 200
    train_h = 80
    train_x = np.zeros((len(extracted_data), train_w * train_h), dtype=np.float32)
    train_y = np.zeros((len(extracted_data)), dtype=np.int64)

    thresh_W = 310
    thresh_H = 150
    for idx in range(len(extracted_data)):
        img = extracted_data[idx]
        H = img.shape[0]
        W = img.shape[1]
        w_pad = 0
        h_pad = 0
        if W < thresh_W:
            w_pad = thresh_W - W
        if H < thresh_H:
            h_pad = (thresh_H - H) // 2

        resized_img = np.pad(img, ((h_pad, h_pad), (0, w_pad)))

        resized_img = cv2.resize(resized_img, (train_w, train_h))

        data = np.divide(resized_img, 255).astype("float32")
        data = np.reshape(data, train_w * train_h)
        train_x[idx, :] = data
        train_y[idx] = extracted_labels[idx]

        # plt.subplot(2, 1, 1)
        # plt.imshow(resized_img)
        # plt.title("Label: " + str(extracted_labels[idx]))
        # plt.subplot(2, 1, 2)
        # plt.imshow(img)
        # plt.title("ORIG")
        # plt.show()

    val_cnt = 70
    idxs = np.arange(len(train_y))
    np.random.shuffle(idxs)

    val_idxs = idxs[:val_cnt]
    val_X = train_x[val_idxs]
    val_Y = train_y[val_idxs]
    dataset = {"train_X": train_x, "train_Y": train_y, "val_X": val_X, "val_Y": val_Y}
    with open("train_dataset_rus_text_score.bin", "wb") as file:
        pickle.dump(dataset, file)


def save_train_data_from_col6():
    # num_examples = len(img_paths)
    # test_X = np.zeros((num_examples, 100 * 100), dtype=np.float32)
    # test_Y = np.zeros((num_examples), dtype=np.int64)
    dir_path = r"Datasets/Score_RusText_Dataset"
    img_paths = np.array(os.listdir(dir_path))
    dataset = None
    # load context
    try:
        with open("data_rus_text_score.bin", "rb") as file:
            dataset = pickle.load(file)
    except:
        # init values
        dataset = {"data": [], "labels": [], "current_idx": 0, "img_paths": img_paths}
        with open("data_rus_text_score.bin", "wb") as file:
            pickle.dump(dataset, file)

    if dataset is not None:
        current_idx = dataset["current_idx"]
        extracted_data = dataset["data"]
        extracted_labels = dataset["labels"]
        img_paths = dataset["img_paths"]
    else:
        current_idx = 0
        extracted_data = []
        extracted_labels = []

    if current_idx >= len(img_paths):
        print("Данных больше нет.")
        return
    else:
        skipped_paths = img_paths[current_idx:]
        for img_path in skipped_paths:
            result_path = os.path.join(dir_path, img_path)
            img = cv2.imread(result_path)
            segmented_img = segment_rus_text_score_col6(img)

            if segmented_img is not None:
                # processed = cv2.resize(segmented_img, (200, 80), cv2.INTER_NEAREST)
                plt.subplot(2, 2, 1)
                plt.imshow(img)
                plt.title('Orig')
                plt.subplot(2, 2, 2)
                plt.title('Segmented')
                plt.imshow(segmented_img)
                plt.show()

                res = input("Введи класс от 0 до 9: \n")
                if res == 'stop':
                    break
                res = int(res)
                #
                if (res < 0 or res > 9) and res != 322:
                    print("Сессия закончена. Гуд лак")
                    # save context info

                    exit()
                if res != 322:
                    extracted_data.append(segmented_img)
                    extracted_labels.append(res)
                    current_idx += 1

                    # save data
                    dataset = {
                        "data": extracted_data, "labels": extracted_labels,
                        "current_idx": current_idx, "img_paths": img_paths
                    }
                    with open("data_rus_text_score.bin", "wb") as file:
                        pickle.dump(dataset, file)

                    print("Количество добавленных: " + str(current_idx))
            else:
                print("Пропущено изображение " + str(img_path))
                continue

    # dataset = {"test_X": test_X, "test_Y": test_Y}
    # with open("test_date_numbers_" + str(num_examples) + "examples_alt.bin", "wb") as file:
    #     pickle.dump(dataset, file)


# save_train_data_from_col6()
process_train_data_from_col6()
# with open("data_rus_text_score.bin", "rb") as file:
#     dataset = pickle.load(file)
#
# current_idx = dataset["current_idx"]
# extracted_data = dataset["data"]
# extracted_labels = dataset["labels"]
# img_paths = dataset["img_paths"]
#
# plt.imshow(extracted_data[-2])
# plt.title("Label: " + str(extracted_labels[-2]))
# plt.show()
