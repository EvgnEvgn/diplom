import os

TEMPLATE_IMG_PATH_P1 = r"InputData\TemplateData\part1.jpg"
TEMPLATE_IMG_PATH_P2 = r"InputData\TemplateData\part2.jpg"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "neural_nets", "models")
RECOGNITION_RESULT_DIR = os.path.join(ROOT_DIR, "Output", "RecognitionResult")
NIST_TOUCHING_DIGITS_DIR = os.path.join(ROOT_DIR, "Datasets", "NIST_Touching_Digits")
NIST_TOUCHING_DIGITS_FILELIST_DIR = os.path.join(NIST_TOUCHING_DIGITS_DIR, "filelists")
NIST_TOUCHING_DIGITS_2D = os.path.join(NIST_TOUCHING_DIGITS_FILELIST_DIR, "2digit")
NIST_TOUCHING_DIGITS_3D = os.path.join(NIST_TOUCHING_DIGITS_FILELIST_DIR, "3digit")

NIST_TOUCHING_DIGITS_2D_TRAIN = os.path.join(NIST_TOUCHING_DIGITS_2D, "train.txt")
NIST_TOUCHING_DIGITS_2D_TEST = os.path.join(NIST_TOUCHING_DIGITS_2D, "test.txt")
NIST_TOUCHING_DIGITS_2D_VAL = os.path.join(NIST_TOUCHING_DIGITS_2D, "val.txt")

NIST_TOUCHING_DIGITS_3D_TRAIN = os.path.join(NIST_TOUCHING_DIGITS_3D, "train.txt")
NIST_TOUCHING_DIGITS_3D_TEST = os.path.join(NIST_TOUCHING_DIGITS_3D, "test.txt")
NIST_TOUCHING_DIGITS_3D_VAL = os.path.join(NIST_TOUCHING_DIGITS_3D, "val.txt")

NIST_TOUCHING_DIGITS_3D = os.path.join(NIST_TOUCHING_DIGITS_FILELIST_DIR, "3digit")
NIST_SD19_ISOLATED_DIGITS_DIR = os.path.join(ROOT_DIR, "Datasets", "NIST_SD19_ISOLATE_DIGITS")
NIST_TOUCHING_DIGITS_SRC = "C:\\Users\\User\\ToucningDigitsDataset\\SyntheticDigitStrings"
NIST_SD19_ISOLATED_DIGITS_SRC = "F:\\DIPLOM\\Single_handwritten_digits"

nist_sd19_isolate_digits_train_data_64 = os.path.join(NIST_SD19_ISOLATED_DIGITS_DIR, "train_data_64.bin")
digits_recognition_col1_data = os.path.join(RECOGNITION_RESULT_DIR, "digits_recognition_col1")

NIST_SD19_ISOLATED_DIGITS_TRAIN_FOLDERS = "hsf_0,hsf_1,hsf_2,hsf_3"
NIST_SD19_ISOLATED_DIGITS_VAL_AND_TEST_FOLDER = "hsf_7"


digits_two_layer_net_model = "digits2layerNN.nnm"
digits_fully_connected_net_model = "digitsFCNN.nnm"
letters_fully_connected_net_model = "lettersFCNN.nnm"
letters_three_layer_cnn_model = "letters-3L-CNN.nnm"
digits_CNN_tf = "digits-CNN-tf.nnm"
digits_CNN_DataGen_tf = "digits-CNN-DG-tf-nnm.h5"
letters_CNN_tf = "letters-CNN-tf.nnm"
letters_CNN_DataGen_tf = "letters-CNN-DG-tf-nnm.h5"
eng_letters_tf = "eng_letters_tf_model.h5"
eng_uppercase_letters_tf = "eng_uppercase_letters_tf_model.h5"
eng_uppercase_letters64_tf = "eng_uppercase_letters64_tf_model.h5"
eng_uppercase_letters64_v2_tf = "eng_uppercase_letters64_v2_tf_model.h5"
touching_dgts_L_classifier_tf_model = "touching_dgts_tf_model.h5"
touching_dgts_C2_classifier_tf_model = "touching_2dgts_classfier_10epochs_BN_tf_model.h5"

