# Import libraries
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
import os
from align_images import align_images_v2
import config


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def save_in_JPG_by_pdf_2_image(path_to_pdf_file, path_to_folder, common_filename='page_'):
    # Path of the pdf
    PDF_file = path_to_pdf_file

    # Store all the pages of the PDF in a variable
    pages = convert_from_path(PDF_file, dpi=300, thread_count=5, first_page=0, last_page=2)

    # Counter to store images of each page of PDF to image
    image_counter = 1

    # Iterate through all the pages stored above
    for page in pages:
        filename = os.path.join(path_to_folder, common_filename + str(image_counter) + ".jpg")

        # Save the image of the page in system
        page.save(filename, 'JPEG')

        # Increment the counter to update filename
        image_counter = image_counter + 1


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def detect_boxes_v2(img_path, result_path):
    # img = cv2.imread(img_path, 0)
    # (thresh, img_bin) = cv2.threshold(img, 128, 255,
    #                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    # img_bin = 255 - img_bin  # Invert the image
    #
    # cv2.imwrite(os.path.join(result_path, "Image_bin.jpg"), img_bin)
    #
    # contours, hierarchy = cv2.findContours(
    #     img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # Sort all the contours by top to bottom.
    # (contours, _) = sort_contours(contours, method="top-to-bottom")
    #
    # for idx, c in enumerate(contours):
    #     # Returns the location and width,height for every contour
    #     x, y, w, h = cv2.boundingRect(c)
    #
    #     # if w > 3000 and h > 1400: #для первой страницы ведомости определяем часть таблицы
    #     idx += 1
    #     new_img = img[y:y + h, x:x + w]
    #     cropped_result_filename_path = os.path.join(result_path, "CroppedImages", str(idx) + '.jpg')
    #     cv2.imwrite(cropped_result_filename_path, new_img)

    img = cv2.imread(img_path, 0)
    (thresh, img_bin) = cv2.threshold(img, 250, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image

    cpy = img_bin.copy()
    morph_size = (8, 8)

    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    img_bin = ~cpy
    cv2.imwrite(os.path.join(result_path, "img_bin_1.jpg"), img_bin)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    min_text_height_limit = 10
    max_text_height_limit = 50

    for contour in contours:
        box = cv2.boundingRect(contour)
        h = box[3]
        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)

    rows = {}
    cols = {}
    cell_threshold = 20
    min_columns = 5

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # Filtering out the clusters having less than 2 cols
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        hor_lines.append((x, y, max_x, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        ver_lines.append((x, y, x, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x, max_y, max_x, max_y))
    vis = img.copy()

    for line in hor_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for line in ver_lines:
        [x1, y1, x2, y2] = line
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imwrite(os.path.join(result_path, "vis_result.jpg"), vis)


def detect_boxes(img_path, result_path):
    img = cv2.imread(img_path, 0)
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255 - img_bin  # Invert the image
    cv2.imwrite(os.path.join(result_path, "Image_bin.jpg"), img_bin)

    # Defining a kernel length
    kernel_length_verti = np.array(img).shape[1] // 140
    kernel_length_hori = np.array(img).shape[1] // 40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_verti))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_hori, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite(os.path.join(result_path, "verticle_lines.jpg"), verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite(os.path.join(result_path, "horizontal_lines.jpg"), horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha

    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=3)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite(os.path.join(result_path, "img_final_bin.jpg"), img_final_bin)
    # Find contours for image, which will detect all the boxes

    contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, _) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        # If the box height is greater then 20, width is >80, then only save it as a box in "cropped/" folder.
        # if (w > 80 and h > 20) and w > 3 * h:
        idx += 1
        new_img = img[y:y + h, x:x + w]
        cropped_result_filename_path = os.path.join(result_path, "CroppedImages", str(idx) + '.jpg')
        cv2.imwrite(cropped_result_filename_path, new_img)


def detect_main_table_box(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin  # Invert the image

    kernel_length_verti = np.array(img).shape[1] // 140
    kernel_length_hori = np.array(img).shape[1] // 40

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_verti))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_hori, 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    alpha = 0.5
    beta = 1.0 - alpha

    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=3)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_of_main_table = None

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 3200 and 1350 < h < 1500:
            img_of_main_table = img[y:y + h, x:x + w]
            break

    return img_of_main_table


def extract_main_table(img_path, result_path):
    img = cv2.imread(img_path)
    template_img = cv2.imread(config.TEMPLATE_IMG_PATH_P1)

    aligned_img = align_images_v2(img, template_img)

    main_table_box = detect_main_table_box(aligned_img)

    cv2.imwrite(os.path.join(result_path, "main_table.jpg"), main_table_box)



def detect_boxes_v3(img_path, result_path):
    img = cv2.imread(img_path, 0)

    ret, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    bitwise = cv2.bitwise_not(thresh1)
    erosion = cv2.erode(bitwise, np.ones((3, 1), np.uint8), iterations=10)
    dilation = cv2.dilate(erosion, np.ones((1, 250), np.uint8), iterations=3)
    cv2.imwrite(os.path.join(result_path, "dilation.jpg"), dilation)

    # (thresh, img_final_bin) = cv2.threshold(dilation, 240, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imwrite(os.path.join(result_path, "img_final_bin.jpg"), img_final_bin)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # If the box height is greater then 20, width is >80, then only save it as a box in "cropped/" folder.
        # if (w > 80 and h > 20) and w > 3 * h:
        idx += 1
        new_img = img[y - 10:y + h + 10, x:x + w]
        cropped_result_filename_path = os.path.join(result_path, "CroppedImages", str(idx) + '.jpg')
        cv2.imwrite(cropped_result_filename_path, new_img)

# для второй части ведомости, отделения основной таблицы от итоговой (в кот приводится статистика оценок студентов)
# erosion = cv2.erode(bitwise, np.ones((2, 2), np.uint8), iterations=4)
# dilation = cv2.dilate(erosion, np.ones((3, 3), np.uint8), iterations=8)
