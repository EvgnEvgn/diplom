import os

TEMPLATE_IMG_PATH_P1 = r"InputData\TemplateData\part1.jpg"
TEMPLATE_IMG_PATH_P2 = r"InputData\TemplateData\part2.jpg"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "neural_nets", "models")

digits_two_layer_net_model = "digits2layerNN.nnm"
digits_fully_connected_net_model = "digitsFCNN.nnm"
letters_fully_connected_net_model = "lettersFCNN.nnm"
letters_three_layer_cnn_model = "letters-3L-CNN.nnm"
digits_CNN_tf = "digits-CNN-tf.nnm"
letters_CNN_tf = "letters-CNN-tf.nnm"
