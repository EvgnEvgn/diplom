# import helpers
import os
import sys
import cv2
from neural_nets.TwoLayerNet.two_layer_net import TwoLayerNet
from neural_nets.TwoLayerNet.two_layer_net_test import find_best_two_layer_net
from neural_nets.FullyConnectedNet.fully_connected_net_test import find_best_digits_fully_connected_model
import mnist
from neural_nets.model_utils import *
from neural_nets.data_utils import *
from align_images import *
import string
from helpers import sort_contours
# import pytesseract
import helpers
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils
from skimage import measure


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


def segment_digits(img_bin):
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)

    # (thresh, img_bin) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_bin)
    # img_bin = cv2.bitwise_not(img_bin)

    img_dilated = cv2.dilate(img_bin, np.ones((4, 2)), iterations=1)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)

    contours, hierarchy = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    idx = 0
    digits = []
    for rect in rects:
        x, y, w, h = rect
        if w > 15 and h > 15:
            digit = img_dilated[y:y + h, x:x + w]
            # digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
            # digit = np.pad(digit, ((4, 4), (4, 4)), "constant")
            # digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            # digit = cv2.dilate(digit, np.ones((1, 1), dtype=np.uint8))

            digits.append(digit)
            # symbol = cv2.dilate(symbol, (3, 3))

            cv2.imwrite(os.path.join("Output/CroppedImages", "symbol" + str(idx) + ".jpg"), digit)
            idx += 1

    return digits


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
            letter = cv2.resize(letter, (20, 20), interpolation=cv2.INTER_AREA)
            letter = np.pad(letter, ((4, 4), (4, 4)), "constant")
            # letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_AREA)
            # letter = cv2.dilate(letter, kernel2)
            letters.append(letter)
            # symbol = cv2.dilate(symbol, (3, 3))

            cv2.imwrite(os.path.join("Output/CroppedImages", "symbol" + str(idx) + ".jpg"), letter)
            idx += 1

    return letters


def clipped_img(img):
    rows = img.shape[0]
    cols = img.shape[1]
    min_x = - 1
    min_y = -1

    max_x = -1
    max_y = -1

    for i in range(0, rows):
        max_in_row = np.max(img[i, :])
        if max_in_row != 0 and min_x == -1:
            min_x = i

        last_ind = i + 1
        max_in_T_row = np.max(img[-last_ind, :])
        if max_in_T_row != 0 and max_x == -1:
            max_x = rows - i

        if min_x != -1 and max_x != -1:
            break

    for i in range(0, cols):
        max_in_col = np.max(img[:, i])
        if max_in_col != 0 and min_y == -1:
            min_y = i

        last_ind = i + 1
        max_in_T_col = np.max(img[:, -last_ind])
        if max_in_T_col != 0 and max_y == -1:
            max_y = cols - i

        if min_y != -1 and max_y != -1:
            break

    clipped = img[min_x: max_x, min_y:max_y]
    return clipped


def is_empty_img(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    _, img_bin = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    img_row_sum = cv2.reduce(img_bin, 1, cv2.REDUCE_AVG).reshape(-1)
    total_sum = np.sum(img_row_sum)

    if total_sum < 100:
        return True
    else:
        return False


def segment_rus_text_score_col6(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)
    max_v = float(grayscale.max())
    min_v = float(grayscale.min())
    x = max_v - ((max_v - min_v) / 2)
    _, img_bin = cv2.threshold(grayscale, x, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_bin)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    dilate = cv2.dilate(img_bin, dilate_kernel, iterations=10)
    cv2.imwrite(os.path.join("Output/CroppedImages", "dilate.jpg"), dilate)


    contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]

    extracted_rus_text = None
    for rect in rects:
        x, y, w, h = rect
        if w > 150 and h > 40:
            segment = img_bin[y:y + h, x:x + w]
            # digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
            # digit = np.pad(digit, ((4, 4), (4, 4)), "constant")
            # digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            # digit = cv2.dilate(digit, np.ones((1, 1), dtype=np.uint8))
            # symbol = cv2.dilate(symbol, (3, 3))

            cv2.imwrite(os.path.join("Output/CroppedImages", "rus_text.jpg"), segment)
            #img_row_sum = cv2.reduce(segment, 1, cv2.REDUCE_AVG).reshape(-1)
            # th_hori = 20
            # H = segment.shape[0]
            # W = segment.shape[1]
            # uppers_hori = [y for y in range(H - 1) if img_row_sum[y] <= th_hori and img_row_sum[y + 1] > th_hori]
            # lowers_hori = [y for y in range(H - 1) if img_row_sum[y] > th_hori and img_row_sum[y + 1] <= th_hori]
            # segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)
            # for x in uppers_hori:
            #     cv2.line(segment, (0, x), (W, x), (255, 0, 0), 1)
            # for x in lowers_hori:
            #     cv2.line(segment, (0, x), (W, x), (0, 255, 0), 1)
            segment = clipped_img(segment)
            extracted_rus_text = segment
            break

    return extracted_rus_text


def segment_date_from_col3(img):
    if is_empty_img(img):
        return None

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)
    max_v = float(grayscale.max())
    min_v = float(grayscale.min())
    x = max_v - ((max_v - min_v) / 2)
    _, img_bin = cv2.threshold(grayscale, x, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_bin)

    rotated = img_bin
    img_row_sum = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)  # np.sum(img_bin,axis=1).tolist()

    th_hori = 2
    H = img_bin.shape[0]
    W = img_bin.shape[1]
    uppers_hori = [y for y in range(H - 1) if img_row_sum[y] <= th_hori and img_row_sum[y + 1] > th_hori]
    lowers_hori = [y for y in range(H - 1) if img_row_sum[y] > th_hori and img_row_sum[y + 1] <= th_hori]

    # rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    # for x in uppers_hori:
    #     cv2.line(rotated, (0, x), (W, x), (255, 0, 0), 1)
    # for x in lowers_hori:
    #     cv2.line(rotated, (0, x), (W, x), (0, 255, 0), 1)
    if len(uppers_hori) == 0 and len(lowers_hori) == 0:
        return None

    if len(uppers_hori) == 0 and len(lowers_hori) != 0:
        uppers_hori.append(0)
    if len(lowers_hori) == 0 and len(uppers_hori) != 0:
        lowers_hori.append(H - 1)
    elif uppers_hori[-1] > lowers_hori[-1]:
        lowers_hori.append(H - 1)

    th_verti = 8
    y_upper = uppers_hori[-1]
    y_lower = lowers_hori[-1]
    new_img = rotated[uppers_hori[-1]:lowers_hori[-1], :]

    cv2.imwrite(os.path.join("Output/CroppedImages", "new_img.jpg"), new_img)

    # rotated = deskew_v3(new_img, new_img.shape[1], new_img.shape[0])
    rotated = new_img
    img_col_sum = cv2.reduce(rotated, 0, cv2.REDUCE_AVG).reshape(-1)
    H = rotated.shape[0]
    W = rotated.shape[1]

    left_verti = [x for x in range(W - 1) if img_col_sum[x] <= th_verti and img_col_sum[x + 1] > th_verti]
    right_verti = [x for x in range(W - 1) if img_col_sum[x] > th_verti and img_col_sum[x + 1] <= th_verti]

    if len(left_verti) == 0:
        return None
    if len(right_verti) == 0:
        return None

    if left_verti[0] >= right_verti[0]:
        left_verti.insert(0, 0)
    if left_verti[-1] > right_verti[-1]:
        right_verti.append(W - 1)

    left_len = len(left_verti)
    right_len = len(right_verti)
    if left_len != right_len:
        print("left_len != right_len. Errors: ")

    new_img = rotated[:, left_verti[0]:right_verti[-1]]
    # new_img = np.pad(new_img, ((2, 2), (2, 2)))
    new_img = cv2.resize(new_img, (220, 60))
    rotated = new_img
    H = rotated.shape[0]
    W = rotated.shape[1]

    img_col_sum = cv2.reduce(rotated, 0, cv2.REDUCE_AVG).reshape(-1)
    img_row_sum = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)
    left_verti = [x for x in range(W - 1) if img_col_sum[x] <= th_verti and img_col_sum[x + 1] > th_verti]
    right_verti = [x for x in range(W - 1) if img_col_sum[x] > th_verti and img_col_sum[x + 1] <= th_verti]

    if len(left_verti) == 0:
        return None

    if len(right_verti) == 0:
        return None
    if left_verti[0] >= right_verti[0]:
        left_verti.insert(0, 0)
    if left_verti[-1] > right_verti[-1]:
        right_verti.append(W - 1)

    left_len = len(left_verti)
    right_len = len(right_verti)
    if left_len != right_len:
        print("left_len != right_len. Errors: ")

    chars_borders = []
    for idx in range(0, len(left_verti)):
        left_border = left_verti[idx]
        right_border = right_verti[idx]

        diff = right_border - left_border
        min_char_thresh = 5
        max_char_thresh = 90
        if diff < min_char_thresh:
            continue
        if diff > max_char_thresh:
            continue
        roi = rotated[:, left_border:right_border]
        sum_by_verti = np.sum(img_col_sum[left_border:right_border])
        sum_by_hori = cv2.reduce(roi, 1, cv2.REDUCE_AVG).reshape(-1)
        first_valued_pixel_in_rows = np.where(sum_by_hori > 5)[0][0]
        thresh_for_hori = roi.shape[0] / 2
        if sum_by_verti < 350:
            continue
        chars_borders.append((left_border, right_border))

    # # clear character borders
    # # find first two characters
    # first_boxes = []
    # first_part_ch = None
    # for l, r in chars_borders:
    #     if l < 60 and r < 90:
    #         first_boxes.append((l, r))
    # length = len(first_boxes)
    # if length > 1:
    #     left = first_boxes[0][0]
    #     right = first_boxes[-1][1]
    #
    #     first_part_ch = rotated[:, left:right]
    #     cv2.imwrite(os.path.join("Output/CroppedImages", "first_part_ch.jpg"), first_part_ch)
    #
    # for l, r in chars_borders:
    #     is_one_ch = r - l < 30
    #     is_two_ch = r - l < 40
    #     is_more_ch = r - l < 65

    # extract characters
    extracted_numbers = []
    idx = 1
    for l, r in chars_borders:
        roi = rotated[:, l:r]

        # roi = cv2.dilate(roi, np.ones((3, 3)), iterations=2)
        # result = cv2.erode(result, np.ones((2, 2)), iterations=1)
        result = cv2.resize(roi, (80, 80), interpolation=cv2.INTER_CUBIC)
        result = np.pad(result, ((10, 10), (10, 10)))

        result = rotate_img(result)
        clipped = clipped_img(result)
        result = np.pad(clipped, ((5, 5), (5, 5)))
        result = cv2.resize(result, (64, 64), interpolation=cv2.INTER_NEAREST)
        # result = deskew_v2(result)
        cv2.imwrite(os.path.join("Output/CroppedImages", "ro1" + str(idx) + ".jpg"), result)
        extracted_numbers.append(result)
        idx += 1

    # rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    # for l, r in chars_borders:
    #     cv2.line(rotated, (l, 0), (l, H), (0, 255, 0), 1)
    #     cv2.line(rotated, (r, 0), (r, H), (255, 0, 0), 1)
    # for x in right_verti:
    #     cv2.line(rotated, (x, 0), (x, H), (255, 0, 0), 1)
    # for x in left_verti:
    #     cv2.line(rotated, (x, 0), (x, H), (0, 255, 0), 1)

    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # # #plt.hist(img_bin.ravel(), bins=261) #range=(0.0, 1.0), fc='k', ec='k')
    # plt.subplot(2, 2, 2)
    # plt.plot(img_col_sum)
    # plt.subplot(2, 2, 3)
    # plt.imshow(img_bin)
    # plt.subplot(2, 2, 4)
    # plt.imshow(rotated)
    # plt.show()

    return extracted_numbers


def segment_touching_digits(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)
    # grayscale = cv2.fastNlMeansDenoising(grayscale)
    # normalized_img = np.zeros_like(grayscale)
    # normalized_img = cv2.normalize(grayscale, normalized_img, 0, 255, cv2.NORM_MINMAX)
    # edged = cv2.Canny(grayscale, 50, 200, 255)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "edged.jpg"), edged)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    thresh, img_bin = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join("Output/CroppedImages", "threshold_img_bin.jpg"), img_bin)

    img_dilated = cv2.dilate(img_bin, dilate_kernel, iterations=7)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)

    img_eroded = cv2.erode(img_dilated, erode_kernel, iterations=2)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_eroded.jpg"), img_eroded)

    # img_dilated = cv2.dilate(img_eroded, kernel, iterations=1)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)
    contours, hierarchy = cv2.findContours(img_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    # rects = imutils.grab_contours(contours)
    idx = 0
    segmented_img = None
    for rect in rects:
        x, y, w, h = rect
        if w > 50 and h > 40:
            ROI = img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join("Output/CroppedImages", "ROI" + str(idx) + ".jpg"), ROI)

            resized = cv2.resize(ROI, (64, 64))
            # resized = np.pad(resized, ((5, 5), (5, 5)))
            # resized = cv2.resize(resized, (64, 64))
            cv2.imwrite(os.path.join("Output/CroppedImages", "resized" + str(idx) + ".jpg"), resized)

            segmented_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            segmented_gray = cv2.GaussianBlur(segmented_gray, (5, 5), 0)

            max_v = float(segmented_gray.max())
            min_v = float(segmented_gray.min())
            x = max_v - ((max_v - min_v) / 2)

            thresh, img_bin = cv2.threshold(segmented_gray, x, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

            # dilated = cv2.dilate(img_bin, np.ones((5, 5)), iterations=3)
            # eroded = cv2.erode(dilated, np.ones((2, 2)), iterations=1)

            clipped = clipped_img(img_bin)

            result = np.pad(clipped, ((4, 4), (4, 4)))
            result = cv2.resize(result, (64, 64))

            cv2.imwrite(os.path.join("Output/CroppedImages", "symbol_dilated" + str(idx) + ".jpg"), result)

            segmented_img = result

    return segmented_img


def watershed_asdas(img, img_bin):
    D = ndimage.distance_transform_edt(img_bin)
    localMax = peak_local_max(D, indices=False, min_distance=45,
                              labels=img_bin)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=img_bin, connectivity=8)

    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    idx = 0
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(img_bin.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(ctr) for ctr in cnts[0]]

        for rect in rects:
            x, y, w, h = rect
            if w > 15 and h > 25:
                cropped = img[y:y + h, x:x + w]
                cv2.imwrite(os.path.join("Output/CroppedImages", "cropped_by_water" + str(idx) + ".jpg"), cropped)
                idx += 1


def compute_skew(img):
    # load in grayscale:
    src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    height, width = src.shape[0:2]

    # invert the colors of our image:
    cv2.bitwise_not(src, src)

    # Hough transform:
    minLineLength = width / 2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    # calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)

    for line in lines:
        angle += math.atan2(line[0][3] * 1.0 - line[0][1] * 1.0, line[0][2] * 1.0 - line[0][0] * 1.0)

    angle /= nb_lines * 1.0

    return angle * 180.0 / np.pi


def deskew(img, angle):
    # load in grayscale:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # invert the colors of our image:
    cv2.bitwise_not(img, img)

    # compute the minimum bounding box:
    non_zero_pixels = cv2.findNonZero(img)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)

    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = img.shape
    rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    # Border removing:
    sizex = np.int0(wh[0])
    sizey = np.int0(wh[1])
    if theta > -45:
        temp = sizex
        sizex = sizey
        sizey = temp
    return cv2.getRectSubPix(rotated, (sizey, sizex), center)


def rotate_img(img_bin, with_morph_op=False):
    img_to_rotate = img_bin

    if with_morph_op:
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        erode = cv2.erode(img_bin, erode_kernel, iterations=1)
        dilate = cv2.dilate(erode, dilate_kernel, iterations=2)

        img_to_rotate = dilate  # cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    rotated = unshear(img_to_rotate)
    # rotated = cv2.erode(rotated, erode_kernel, iterations=1)

    return rotated


def segment_eng_letters(img):
    # TODO: сделать проверку, что в результате сегментации было определено 2 буквы или 1.
    # TODO: В случае если была определена 1 буква, то заново считать и выделить ее (нужно для того, чтобы лучше
    #  предобработать данные)
    # TODO: для 2ух сегментированных букв оставить, как есть

    cv2.imwrite(os.path.join("Output/CroppedImages", "orig.jpg"), img)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    grayscale = cv2.fastNlMeansDenoising(grayscale)
    normalized_img = np.zeros_like(grayscale)
    normalized_img = cv2.normalize(grayscale, normalized_img, 0, 255, cv2.NORM_MINMAX)

    _, img_bin = cv2.threshold(normalized_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_bin)

    rotated = rotate_img(img_bin, with_morph_op=True)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "rotated.jpg"), rotated)

    labels = measure.label(rotated, connectivity=2, background=0)
    # # charCandidates = np.zeros(img_bin.shape, dtype="uint8")
    idx = 0
    cropped_letters = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(img_bin.shape, dtype="uint8")
        labelMask[labels == label] = 255
        cnts, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in cnts]

        for rect in rects:
            x, y, w, h = rect
            if w > 15 and h > 30:
                y_pad = min(y, 1)
                x_pad = min(x, 1)
                w_pad = min(img.shape[1], x + w + 1)
                h_pad = min(img.shape[0], y + h + 1)
                letter = rotated[y - y_pad: h_pad, x - x_pad: w_pad]
                cropped_letters.append(letter)
                # cv2.imwrite(os.path.join("Output/CroppedImages", "symbol" + str(idx) + ".jpg"), letter)
                idx += 1
    k = 0
    final_letters = []
    if len(cropped_letters) == 2:
        for cl in cropped_letters:
            # resized = np.pad(cl, ((4, 4), (4, 4)), "constant")
            # resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_AREA)

            resized = cv2.resize(cl, (56, 56), interpolation=cv2.INTER_NEAREST)
            resized = np.pad(resized, ((4, 4), (4, 4)), "constant")

            final_letters.append(resized)
    else:
        segmented = segment_single_letter(img, img_bin)
        final_letters.append(segmented)

    return final_letters


#     cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# contours, _ = cv2.findContours(markers.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# rects = [cv2.boundingRect(ctr) for ctr in contours]
# idx = 0
# for rect in rects:
#     x, y, w, h = rect
#     if w > 15 and h > 25:
#         x, y, w, h = rect
#         # if w > 15 and h > 21:
#         letter = rotated[y:y + h, x:x + w]
#         cv2.imwrite(os.path.join("Output/CroppedImages", "symbol" + str(idx) + ".jpg"), letter)
#         idx += 1

def segment_single_letter(orig_img, img_bin):
    # rotated_bin = rotate_img(img_bin)
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilate = cv2.dilate(rotated_bin, dilate_kernel, iterations=2)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "dilate" + ".jpg"), dilate)
    #
    # labels = measure.label(dilate, connectivity=2, background=0)
    # # # charCandidates = np.zeros(img_bin.shape, dtype="uint8")
    # idx = 0
    #
    # for label in np.unique(labels):
    #     # if this is the background label, ignore it
    #     if label == 0:
    #         continue
    #
    #     # otherwise, construct the label mask to display only connected components for the
    #     # current label, then find contours in the label mask
    #     labelMask = np.zeros(img_bin.shape, dtype="uint8")
    #     labelMask[labels == label] = 255
    #     cnts, _ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     rects = [cv2.boundingRect(ctr) for ctr in cnts]
    #
    #     for rect in rects:
    #         x, y, w, h = rect
    #         if w > 15 and h > 35:
    #             y_pad = min(y, 5)
    #             x_pad = min(x, 5)
    #             w_pad = min(rotated_bin.shape[1], x + w + 5)
    #             h_pad = min(rotated_bin.shape[0], y + h + 5)
    #             ROI = img_bin[y - y_pad: h_pad, x - x_pad: w_pad]
    #             cv2.imwrite(os.path.join("Output/CroppedImages", "ROI" + str(idx) + ".jpg"), ROI)
    #
    #             cropped_letter = np.zeros((28, 28), dtype=np.uint8)
    #
    #             resized_ROI = cv2.resize(ROI, (20, 20), interpolation=cv2.INTER_AREA)
    #             x = cropped_letter.shape[0] // 2 - resized_ROI.shape[0] // 2
    #             y = cropped_letter.shape[1] // 2 - resized_ROI.shape[1] // 2
    #             cropped_letter[y:y+resized_ROI.shape[1], x:x+resized_ROI.shape[0]] = resized_ROI
    #
    #             cv2.imwrite(os.path.join("Output/CroppedImages", "single_letter_cropped" + str(idx) + ".jpg"), cropped_letter)
    #             idx += 1
    #             #return cropped_letter

    # img_bin = np.pad(img_bin, ((5, 5), (5, 5)))
    # kernel = np.ones((2, 2), np.uint8)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilate = cv2.dilate(img_bin, dilate_kernel, iterations=2)
    cv2.imwrite(os.path.join("Output/CroppedImages", "dilate" + ".jpg"), dilate)

    contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    idx = 0
    segmented_letter = None
    for rect in rects:
        x, y, w, h = rect
        if w > 30 and h > 30:
            x, y, w, h = rect

            y_pad = min(y, 0)
            x_pad = min(x, 0)
            w_pad = min(img_bin.shape[1], x + w + 0)
            h_pad = min(img_bin.shape[0], y + h + 0)
            ROI = img_bin[y - y_pad:h_pad, x - x_pad:w_pad]

            cv2.imwrite(os.path.join("Output/CroppedImages", "ROI" + str(idx) + ".jpg"), ROI)
            ROI = cv2.resize(ROI, (64, 64), interpolation=cv2.INTER_AREA)
            letter = deskew_v2(ROI)
            letter = np.pad(letter, ((10, 10), (10, 10)))
            letter = cv2.resize(letter, (64, 64), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join("Output/CroppedImages", "cropped" + str(idx) + ".jpg"), letter)
            segmented_letter = letter

    return segmented_letter


def deskew_v2(img):
    SZ = 64
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]), flags=affine_flags)
    return img


def correct_skew(image, ):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def deskew_v3(img, W, H):
    delta = 0.05
    limit = 5

    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(img, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = H, W
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, \
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated
    # return rotated


def segment_letters_v2(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, img_bin = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_bin)
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    # dilate_kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # dilate = cv2.dilate(img_bin, dilate_kernel, iterations=1)
    # erode = cv2.erode(dilate, erode_kernel, iterations=1)
    # dilate = cv2.dilate(erode, dilate_kernel_2, iterations=1)
    # img_bin = dilate
    # cv2.imwrite(os.path.join("Output/CroppedImages", "opening.jpg"), img_bin)

    kernel = np.ones((2, 2), np.uint8)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (2, 2))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    erode = cv2.erode(img_bin, erode_kernel, iterations=1)
    dilate = cv2.dilate(erode, dilate_kernel, iterations=2)
    opening = dilate  # cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(os.path.join("Output/CroppedImages", "opening.jpg"), opening)

    edges = cv2.Canny(opening, 50, 150, apertureSize=5)
    cv2.imwrite(os.path.join("Output/CroppedImages", "edges.jpg"), edges)
    watershed_asdas(img, grayscale, edges)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    cv2.imwrite(os.path.join("Output/CroppedImages", "sure_bg.jpg"), sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    cv2.imwrite(os.path.join("Output/CroppedImages", "sure_fg.jpg"), sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    cv2.imwrite(os.path.join("Output/CroppedImages", "unknown.jpg"), unknown)
    cv2.imwrite(os.path.join("Output/CroppedImages", "markers.jpg"), markers)

    img_bin[markers == -1] = 255
    cv2.imwrite(os.path.join("Output/CroppedImages", "marked_img_bin.jpg"), img_bin)
    markers = np.array(markers, dtype=np.uint8)

    # lines = cv2.HoughLinesP(img_bin, rho=1, theta=1 * np.pi / 180, threshold=100, minLineLength=2, maxLineGap=50)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "lines.jpg"), lines)

    # img_dilated = cv2.dilate(sheared_img, np.ones((10, 10)), iterations=1)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)

    # img_eroded = cv2.erode(sheared_img, np.ones((2, 1)), iterations=1)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "img_eroded.jpg"), img_eroded)
    #
    # img_dilated = cv2.dilate(img_eroded, np.ones((2, 1)), iterations=1)
    # cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated_2.jpg"), img_dilated)

    contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in contours]
    idx = 0
    letters = []
    for rect in rects:
        x, y, w, h = rect
        if w > 15 and h > 25:
            x, y, w, h = rect
            # if w > 15 and h > 21:
            letter = img[y:y + h, x:x + w]
            # letter = cv2.resize(letter, (20, 20), interpolation=cv2.INTER_AREA)
            # letter = np.pad(letter, ((4, 4), (4, 4)), "constant")
            # letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_AREA)
            # letter = cv2.dilate(letter, kernel2)
            # letters.append(letter)
            # letter = cv2.dilate(letter, (3, 3))

            cv2.imwrite(os.path.join("Output/CroppedImages", "cropped" + str(idx) + ".jpg"), letter)
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


def segment_digits_from_col1(img):
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
    idx = 0
    for rect in rects:
        x, y, w, h = rect
        # remove dot from boxes
        if w > 5 and h > 5:
            digit = erroded[y:y + h, x:x + w]
            digit = np.pad(digit, ((16, 16), (16, 16)), "constant")
            digit = cv2.resize(digit, (28, 28))
            test = np.pad(img_bin[y:y + h, x:x + w], ((16, 16), (16, 16)), "constant")
            test = cv2.resize(digit, (32, 32))
            cv2.imwrite(os.path.join("Output/CroppedImages", "digit{0}.jpg".format(idx)), test)
            digits.append(digit)
            idx += 1
    return digits


def segment_characters_from_col2(img):
    horizontal_tilt_threshold = 50
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, img_bin) = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_bin.jpg"), img_bin)

    img_dilated = cv2.dilate(img_bin, np.ones((5, 5)), iterations=2)
    cv2.imwrite(os.path.join("Output/CroppedImages", "img_dilated.jpg"), img_dilated)

    contours, hierarchy = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cntrs, _) = sort_contours(contours, method="top-to-bottom")

    rects = np.array([cv2.boundingRect(ctr) for ctr in cntrs])
    y = rects[:, 1]
    steps = abs(y[1:] - y[:-1])
    row_idxs = np.where(steps > horizontal_tilt_threshold)[0]
    row_idxs += 1
    row_idxs = np.insert(row_idxs, 0, 0)
    row_idxs = np.append(row_idxs, len(y))
    rows_count = len(row_idxs) - 1
    idx = 0
    rows = []
    characters = []

    for idx in range(rows_count):
        row = rects[row_idxs[idx]: row_idxs[idx + 1]]
        sorted_by_x = row[row[:, 0].argsort()]
        rows.append(sorted_by_x)

    rows = np.array(rows)

    for row in rows:
        for word in row:
            x, y, w, h = word
            # remove dot from boxes
            if w > 5 and h > 5:
                character = img[y:y + h, x:x + w]
                # character = cv2.resize(character, (32, 32))
                # if w < 140:
                #   character = img_bin[y:y + h, x:x + w]
                #     character = np.pad(character, ((8, 8), (8, 8)), "constant")
                # character = cv2.resize(character, (28, 28))

                cv2.imwrite(os.path.join("Output/CroppedImages", "word" + str(idx) + ".jpg"), character)
                characters.append(character)
                idx += 1

    return characters


def collect_digits_from_col1():
    col_path = r"Output/ExtractedColumns/0_col"
    reshaped_digits = np.zeros((0, 28, 28, 1))
    paths = np.array([])
    for path in os.listdir(col_path):
        full_path = os.path.join(col_path, path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            digits = segment_digits_from_col1(img)
            digits = preprocess_digits_img_for_tf_predictive_model(digits, (len(digits), 28, 28, 1))
            reshaped_digits = np.concatenate((reshaped_digits, digits))
            path_to_digits = [path] * len(digits)
            paths = np.concatenate((paths, path_to_digits))

    return reshaped_digits, paths


def get_letter_images_from_col5():
    col_path = r"Output/ExtractedColumns/4_col"
    listdir = ["{0}.jpg".format(i) for i in range(0, 16)]
    paths = np.array([])
    all_letters = np.zeros((0, 64, 64))
    for path in listdir:
        full_path = os.path.join(col_path, path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            letters = segment_letters(img)
            all_letters = np.concatenate((all_letters, letters))
            path_to_digits = [path] * len(letters)
            paths = np.concatenate((paths, path_to_digits))

    return all_letters, paths


def collect_letters_from_col5():
    col_path = r"Output/ExtractedColumns/4_col"
    reshaped_letters = np.zeros((0, 28, 28, 1))
    listdir = ["{0}.jpg".format(i) for i in range(0, 40)]
    paths = np.array([])
    for path in listdir:
        full_path = os.path.join(col_path, path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            letters = segment_letters(img)
            letters = preprocess_letters_img_for_predictive_model(letters, (len(letters), 28, 28, 1))
            reshaped_letters = np.concatenate((reshaped_letters, letters))
            path_to_digits = [path] * len(letters)
            paths = np.concatenate((paths, path_to_digits))

    return reshaped_letters, paths


def get_letters_img_for_test():
    path = r"Output/eng_letters_with_issues"
    reshaped_letters = np.zeros((0, 28, 28, 1))
    listdir = os.listdir(path)
    paths = np.array([])
    for img_path in listdir:
        full_path = os.path.join(path, img_path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            letters = segment_letters_v2(img)
            letters = preprocess_letters_img_for_predictive_model(letters, (len(letters), 28, 28, 1))
            reshaped_letters = np.concatenate((reshaped_letters, letters))
            path_to_digits = [img_path] * len(letters)
            paths = np.concatenate((paths, path_to_digits))

    return reshaped_letters, paths


def save_labels(labels, filename):
    file = open(os.path.join(config.ROOT_DIR, filename), "wb")
    pickle.dump(labels, file)


def load_labels(filename):
    file_path = os.path.join(config.digits_recognition_col1_data, filename)
    try:
        file = open(file_path, "rb")
        labels = pickle.load(file)
    except FileNotFoundError:
        return None
    return labels


def save_extracted_data_with_labels(extracted_data, filename):
    file = open(os.path.join(config.digits_recognition_col1_data, filename), "wb")
    pickle.dump(extracted_data, file)


def load_extracted_data_with_labels(filename):
    file_path = os.path.join(config.digits_recognition_col1_data, filename)
    try:
        file = open(file_path, "rb")
        extracted_data = pickle.load(file)
    except FileNotFoundError:
        return None
    return extracted_data


def check_model_accuracy_digits_from_col_1(digits, paths):
    true_labels = []
    # keys consists of ascii coded digits
    keys = [i for i in range(48, 58)]
    model = get_digits_cnn_dg_tf_model()
    extracted_data = {}
    true_labels = load_labels("digit_labels_col_1.dat")

    if len(true_labels) == 0:
        for idx, digit in enumerate(digits):

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

    # save_extracted_data_with_labels(extracted_data, "digits_recognition_col_1.dat")
    # save_labels(true_labels, "digit_labels_col_1.dat")

    print(true_labels)
    print(predicted)
    print(accuracy)


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 40)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

# img = cv2.imread("Output/ExtractedColumns/5_col/8.jpg")
# grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(grayscale, 127, 255, 1)[1]
# thresh = np.pad(thresh, 100, pad_with, padder=0)
#
# rotated = deskew_v2(thresh)
# cv2.imwrite(os.path.join("Output/CroppedImages", "rotated.jpg"), rotated)
# sheared_img = unshear(thresh)
# cv2.imwrite(os.path.join("Output/CroppedImages", "sheared_img.jpg"), sheared_img)
#
# ret, thresh_res = cv2.threshold(sheared_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# cv2.imwrite(os.path.join("Output/CroppedImages", "res_deskewed.jpg"), thresh_res)

# segment_digits(sheared_img)
# img = cv2.imread("Output/CroppedImages/symbol0.jpg", 0)
# cv2.imwrite(os.path.join("Output/CroppedImages", "img_in.jpg"), img)
# thresh = img
# #grayscale = cv2.cvtColor()
# #thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imwrite(os.path.join("Output/CroppedImages", "thresh_in.jpg"), thresh)
# D = ndimage.distance_transform_edt(thresh)
# localMax = peak_local_max(D, indices=False, min_distance=20,
#                           labels=thresh)
#
# markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
# labels = watershed(-D, markers, mask=thresh)
#
# for label in np.unique(labels):
#     # if the label is zero, we are examining the 'background'
#     # so simply ignore it
#     if label == 0:
#         continue
#     # otherwise, allocate memory for the label region and draw
#     # it on the mask
#     mask = np.zeros(img.shape, dtype="uint8")
#     mask[labels == label] = 255
#     cv2.imwrite(os.path.join("Output/CroppedImages", "mask.jpg"), mask)
#     # detect contours in the mask and grab the largest one
#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#                             cv2.CHAIN_APPROX_SIMPLE)
#
#     cnts = imutils.grab_contours(cnts)
#
#     c = max(cnts, key=cv2.contourArea)
#
#     # draw a circle enclosing the object
#     ((x, y), r) = cv2.minEnclosingCircle(c)
#
#     cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
#     cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
# cv2.imshow("Output", img)
# cv2.waitKey(0)
