import helpers
import os
import cv2
import numpy as np
from align_images import alignImages, correct_table_skew
from extracting_data import extract_data_from_main_table
from neural_nets.model_utils import *
import emnist
import matplotlib.pyplot as plt
import cv2

#input_path = r"InputData/Data"
#output_path = r"Output"
#extract_data_from_main_table(input_path, output_path)

input_path = r"InputData/Data"
output_path = r"Output"
extract_data_from_main_table(input_path, output_path)
#
# print(cv2.__version__)

# x_train, y_train, x_val, y_val, x_test, y_test = get_emnist_letters_data_TF()
#
# fcmodel = get_letters_fully_connected_net_model()
# print(fcmodel.train_accuracy)
# print(fcmodel.test_accuracy)
#
# tf_cnn_model = get_letters_CNN_tf_model()
# train_accuracy = tf_cnn_model.predict(x_train, verbose=0)
# test_accuracy = tf_cnn_model.evaluate(x_test, y_test, verbose=0)
# print(train_accuracy)
# print(test_accuracy)
#
# tf_cnn_dg_model = get_letters_cnn_dg_tf_model()
# train_accuracy = tf_cnn_dg_model.evaluate(x_train, y_train, verbose=0)
# test_accuracy = tf_cnn_dg_model.evaluate(x_test, y_test, verbose=0)
# print(train_accuracy)
# print(test_accuracy)

print(emnist.list_datasets())
