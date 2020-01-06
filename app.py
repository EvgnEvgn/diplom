import helpers
import os
import cv2
import numpy as np
from align_images import alignImages, correct_table_skew
from extracting_data import extract_data_from_main_table


def save_align_images_result():
    template_img_path = r"C:\Users\User\PycharmProjects\diplom\InputData\TemplateData\part2_1.jpg"
    processing_img_path = r"C:\Users\User\PycharmProjects\diplom\InputData\data\page_2.jpg"
    template_img = cv2.imread(template_img_path, cv2.IMREAD_COLOR)
    processing_img = cv2.imread(processing_img_path, cv2.IMREAD_COLOR)
    result_img, h = alignImages(template_img, processing_img)
    # Write aligned image to disk.
    outFilename = r"C:\Users\User\PycharmProjects\diplom\Output\aligned.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, result_img)
    # Print estimated homography
    print("Estimated homography : \n", h)


# input_path = r"InputData\Data\page_1.jpg"
# output_path = r"Output"
#
#main_table_img_p2 = helpers.extract_main_table_part2(input_path, output_path)
#helpers.extract_table_rows(main_table_img_p2, output_path)
# img = cv2.imread(input_path)
# cv2.imwrite(os.path.join(output_path, "img.jpg"), img)
# corrected_img = correct_table_skew(img)
# cv2.imwrite(os.path.join(output_path, "correction_of_skew.jpg"), corrected_img)
# input_path = r"InputData\Data\page_"
# output_path = r"Output\ExtractingMainTable"
# idxs = np.array(range(1, 21))
# filename_ext = ".jpg"
# odd_idxs = idxs[idxs % 2 == 0]
# for idx in odd_idxs:
#     filename_output_path = os.path.join(output_path, "main_table_p2_" + str(idx) + filename_ext)
#     filename_input_path = input_path + str(idx) + filename_ext
#     helpers.extract_main_table_part2(filename_input_path, filename_output_path)


input_path = r"InputData/Data"
output_path = r"Output"
extract_data_from_main_table(input_path, output_path)
