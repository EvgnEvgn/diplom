import helpers
import os
import sys
import numpy as np
import cv2
from neural_nets.TwoLayerNet.two_layer_net import TwoLayerNet
from neural_nets.TwoLayerNet.two_layer_net_test import find_best_two_layer_net
from neural_nets.FullyConnectedNet.fully_connected_net_test import find_best_digits_fully_connected_model
import mnist
import matplotlib.pyplot as plt
from neural_nets.model_utils import *
from neural_nets.data_utils import *
from align_images import *
import string
from helpers import sort_contours


def extract_data_from_main_table(input_path, output_path):
    table_boxes = None

    for path in os.listdir(input_path):
        full_path = os.path.join(input_path, path)
        if os.path.isfile(full_path):
            filename = path.split(".")[0]
            page_number = int(filename.split("_")[1])
            if page_number % 2 != 0:
                table_box = helpers.extract_main_table_part1(full_path, output_path)
            else:
                table_box = helpers.extract_main_table_part2(full_path, output_path)

            rows_imgs = np.array(helpers.extract_table_rows(table_box, output_path))
            if table_boxes is None:
                table_boxes = rows_imgs
            else:
                table_boxes = np.concatenate((table_boxes, rows_imgs), axis=0)

    idx = 0
    for row in table_boxes:
        for col_idx in range(len(row)):
            path_to_col_folder = os.path.join(output_path, "ExtractedColumns", "{0}_col".format(col_idx))
            if not os.path.exists(path_to_col_folder):
                os.mkdir(path_to_col_folder)

            cropped_result_filename_path = os.path.join(path_to_col_folder, "{0}.jpg".format(idx))
            cv2.imwrite(cropped_result_filename_path, row[col_idx])

        idx += 1
    return


def segment_digits(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)

    (thresh, img_bin) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY_INV)
    # img_bin = cv2.bitwise_not(img_bin)

    img_dilated = cv2.dilate(img_bin, np.ones((2, 2)), iterations=5)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_dilated)

    contours, hierarchy = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    idx = 0
    digits = []
    for rect in rects:
        x, y, w, h = rect
        digit = img_bin[y:y + h, x:x + w]
        digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
        digit = np.pad(digit, ((4, 4), (4, 4)), "constant")
        # digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit = cv2.dilate(digit, np.ones((1, 1), dtype=np.uint8))
        digits.append(digit)
        # symbol = cv2.dilate(symbol, (3, 3))

        cv2.imwrite(os.path.join("Output/CroppedImages", "symbol" + str(idx) + ".jpg"), digit)
        idx += 1

    return digits


def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape

    edges = cv2.Canny(image, 150, 200, 3, 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=width / 2.0, maxLineGap=20)
    angle = 0.0
    nlines = lines.size
    for x1, y1, x2, y2 in lines[0]:
        angle += np.arctan2(y2 - y1, x2 - x1)
    return angle / nlines


def deskew(image, angle):
    image = cv2.bitwise_not(image)
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    return cv2.getRectSubPix(rotated, (cols, rows), center)


def deskew_v2(img, size):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11'] / m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5 * size * skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (size, size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def segment_letters(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "gaussian_blur.jpg"), grayscale)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    (thresh, img_bin) = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join("Output/CroppedImages", "threshold_img_bin.jpg"), img_bin)

    img_dilated = cv2.dilate(img_bin, dilate_kernel, iterations=2)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)

    img_eroded = cv2.erode(img_dilated, erode_kernel, iterations=2)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_eroded.jpg"), img_eroded)

    # img_dilated = cv2.dilate(img_eroded, kernel, iterations=1)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)
    contours, hierarchy = cv2.findContours(img_eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    idx = 0
    letters = []
    for rect in rects:
        x, y, w, h = rect
        if w > 20 and h > 10:
            letter = img_eroded[y:y + h, x:x + w]
            # letter = cv2.resize(letter, (20, 20), interpolation=cv2.INTER_AREA)
            # letter = np.pad(letter, ((4, 4), (4, 4)), "constant")
            letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_AREA)
            letter = cv2.dilate(letter, kernel2)
            letters.append(letter)
            # symbol = cv2.dilate(symbol, (3, 3))

            cv2.imwrite(os.path.join("Output/CroppedImages", "symbol" + str(idx) + ".jpg"), letter)
            idx += 1

    return letters


def process_orig_digit_img(orig_digits_img):
    scaled_digits = np.divide(orig_digits_img, 255)
    one_dim_digits = scaled_digits.reshape(len(orig_digits_img), -1)

    return one_dim_digits


def predict_digits():
    img = cv2.imread("Output/ExtractedColumns/5_col/15.jpg")

    digits = segment_digits(img)
    digits = process_orig_digit_img(digits)

    neural_net = get_digits_fully_connected_net_model()
    predicted = np.argmax(neural_net.loss(digits), axis=1)

    print(predicted)


def predict_letters_using_tf():
    alphabet = dict((idx, key) for idx, key in enumerate(string.ascii_lowercase))
    img = cv2.imread("Output/ExtractedColumns/4_col/193.jpg")
    letters = segment_letters(img)
    # x_train, y_train, x_val, y_val, x_test, y_test = get_emnist_letters_data_TF()
    letters = preprocess_letters_img_for_predictive_model(letters, model_input_shape=(len(letters), 28, 28, 1))

    letters_model = get_letters_CNN_tf_model()
    predicted = letters_model.predict_classes(letters)
    print(alphabet[predicted[0]])

    # score = letters_model.evaluate(x_test, y_test, verbose=0)
    # print(score)


def segment_digits_from_column_1(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)

    (thresh, img_bin) = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_bin)

    img_dilated = cv2.dilate(img_bin, np.ones((7, 1)), iterations=1)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)

    erroded = cv2.erode(img_dilated, (10, 10), iterations=1)
    cv2.imwrite(os.path.join("Output/CroppedImages", "erroded.jpg"), erroded)

    contours, hierarchy = cv2.findContours(erroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (contours, _) = sort_contours(contours)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    digits = []
    for rect in rects:
        x, y, w, h = rect
        # remove dot from boxes
        if w > 5 and h > 5:
            digit = erroded[y:y + h, x:x + w]
            digit = np.pad(digit, ((16, 16), (16, 16)), "constant")
            digit = cv2.resize(digit, (28, 28))

            digits.append(digit)

    return digits


def collect_digits_from_col_1():
    col_path = r"Output/ExtractedColumns/0_col"
    reshaped_digits = np.zeros((0, 28, 28, 1))
    paths = np.array([])
    for path in os.listdir(col_path):
        full_path = os.path.join(col_path, path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            digits = segment_digits_from_column_1(img)
            digits = preprocess_digits_img_for_tf_predictive_model(digits, (len(digits), 28, 28, 1))
            reshaped_digits = np.concatenate((reshaped_digits, digits))
            path_to_digits = [path] * len(digits)
            paths = np.concatenate((paths, path_to_digits))

    return reshaped_digits, paths


def save_labels(labels, filename):
    file = open(os.path.join(config.ROOT_DIR, filename), "wb")
    pickle.dump(labels, file)


def load_labels(filename):
    file_path = os.path.join(config.ROOT_DIR, filename)
    try:
        file = open(file_path, "rb")
        labels = pickle.load(file)
    except FileNotFoundError:
        return None
    return labels


def save_extracted_data_with_labels(extracted_data, filename):
    file = open(os.path.join(config.ROOT_DIR, filename), "wb")
    pickle.dump(extracted_data, file)


def load_extracted_data_with_labels(filename):
    file_path = os.path.join(config.ROOT_DIR, filename)
    try:
        file = open(file_path, "rb")
        extracted_data = pickle.load(file)
    except FileNotFoundError:
        return None
    return extracted_data


def check_model_accuracy_digits_from_col_1(digits, paths):
    true_labels = []
    keys = [i for i in range(48, 58)]
    model = get_digits_cnn_dg_tf_model()
    extracted_data = {}
    # true_labels = load_labels("digit_labels_col_1.dat")

    digits = digits[:50]

    if len(true_labels) == 0:
        for idx, digit in enumerate(digits):
            if idx == 12 or idx == 21:
                cv2.imshow('{0} - type digit'.format(paths[idx]), digit)
                key = cv2.waitKey(0)

                if key == 27:  # (escape to quit)
                    # TODO:Save true data
                    sys.exit()
                elif key in keys:
                    true_labels.append(int(chr(key)))
                cv2.destroyAllWindows()

    predicted = model.predict_classes(digits)
    accuracy = np.mean(true_labels == predicted)
    extracted_data["digits"] = digits
    extracted_data["labels"] = true_labels
    extracted_data["predicted"] = predicted
    extracted_data["accuracy"] = accuracy
    #
    save_extracted_data_with_labels(extracted_data, "digits_recognition_col_1.dat")
    save_labels(true_labels, "digit_labels_col_1.dat")

    print(true_labels)
    print(predicted)
    print(accuracy)


# img = cv2.imread("Output/ExtractedColumns/0_col/24.jpg")
# digits = segment_digits_from_column_1(img)
# digits = preprocess_digits_img_for_tf_predictive_model(digits, (len(digits), 28, 28, 1))
# model = get_digits_cnn_dg_tf_model()
# predicted = model.predict_classes(digits)
# print(predicted)
digits, paths = collect_digits_from_col_1()
check_model_accuracy_digits_from_col_1(digits, paths)
# extracted_digits_data = load_extracted_data_with_labels("digits_recognition_col_1.dat")
# count = len(extracted_digits_data["predicted"])
# print(np.where(~np.equal(extracted_digits_data["predicted"], extracted_digits_data["labels"])))