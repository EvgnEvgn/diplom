import helpers
import os
import numpy as np
import cv2
from neural_nets.TwoLayerNet.two_layer_net import TwoLayerNet
from neural_nets.TwoLayerNet.two_layer_net_test import find_best_net
import mnist
import matplotlib.pyplot as plt


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

    img_dilated = cv2.dilate(img_bin, None, iterations=2)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_dilated)

    contours, hierarchy = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    idx = 0
    digits = []
    for rect in rects:
        # Draw the rectangles
        # cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        # leng = int(rect[3] * 1.6)
        # pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        # pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        # roi = img_bin[pt1:pt1 + leng, pt2:pt2 + leng]

        x, y, w, h = rect
        symbol = img_bin[y:y + h, x:x + w]
        symbol = cv2.resize(symbol, (20, 20), interpolation=cv2.INTER_BITS)
        symbol = np.pad(symbol, ((4, 4), (4, 4)), "constant")
        symbol = cv2.dilate(symbol, np.ones((2,2), dtype=np.uint8), iterations=1)
        digits.append(symbol)
        # symbol = cv2.dilate(symbol, (3, 3))

        cv2.imwrite(os.path.join("Output/CroppedImages", "symbol" + str(idx) + ".jpg"), symbol)
        idx += 1
        # Resize the image
        # roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        # nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        # cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    return digits


img = cv2.imread("Output/ExtractedColumns/5_col/11.jpg")

digits = segment_digits(img)
digits = np.array(digits)
plt.subplot(2,1,1)
plt.hist(digits[0])

mnist_digits = mnist.test_images()[:10]
labels = mnist.test_labels()[:10]
plt.subplot(2,1,2)
plt.hist(mnist_digits[6])

plt.show()
# digits = np.array(digits)
# neural_net = find_best_net()
# predicted = np.argmax(neural_net.loss(digits), axis=1)
# print(predicted)
