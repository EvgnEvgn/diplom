import helpers
import os
import cv2

from align_images import alignImages


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


template_input_path = r"InputData\TemplateData\part1_1.jpg"
input_path = r"InputData\Data\page_1.jpg"
output_path = r"Output"
# img = cv2.imread(input_path, 0)
#helpers.detect_boxes(input_path, output_path)
#helpers.detect_boxes_v3(input_path, output_path)
#align_images_v2(template_input_path, template_input_path, output_path)
helpers.extract_main_table(input_path, output_path)
