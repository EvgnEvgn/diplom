import helpers
import os
import numpy as np
import cv2


def extract_data(input_path, output_path):
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
