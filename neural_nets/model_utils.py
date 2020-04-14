import config
import pickle
import os
from neural_nets.TwoLayerNet.two_layer_net_test import find_best_two_layer_net
from neural_nets.FullyConnectedNet.fully_connected_net_test import *
from neural_nets.CNN.three_layerCNN_test import *
from neural_nets.data_utils import *
from tensorflow.keras.models import load_model


def save_best_model(best_model, filename):
    file_path = os.path.join(config.MODELS_DIR, filename)
    file = open(file_path, "wb")

    pickle.dump(best_model, file)


def load_best_model(filename):
    file_path = os.path.join(config.MODELS_DIR, filename)
    try:
        file = open(file_path, "rb")
        best_model = pickle.load(file)
    except FileNotFoundError:
        return None
    return best_model


def get_digits_two_layer_net_model():
    model = load_best_model(config.digits_two_layer_net_model)

    if model is None:
        digits_mnist_data = get_mnist_digits_data()
        model = find_best_two_layer_net(digits_mnist_data)
        save_best_model(model, config.digits_two_layer_net_model)

    return model


def get_digits_fully_connected_net_model():
    model = load_best_model(config.digits_fully_connected_net_model)

    if model is None:
        digits_mnist_data = get_mnist_digits_data()
        model = find_best_digits_fully_connected_model(digits_mnist_data)
        save_best_model(model, config.digits_fully_connected_net_model)

    return model


def get_letters_fully_connected_net_model():
    model = load_best_model(config.letters_fully_connected_net_model)

    if model is None:
        letters_emnist_data = get_emnist_letters_data()
        model = find_best_letters_fully_connected_model(letters_emnist_data)
        save_best_model(model, config.letters_fully_connected_net_model)

    return model


def get_letters_three_layer_cnn_model():
    model = load_best_model(config.letters_three_layer_cnn_model)

    if model is None:
        letters_emnist_data = get_emnist_letters_data_for_CNN(num_training=100)
        model = find_best_letters_three_layer_CNN_model(letters_emnist_data)
        save_best_model(model, config.letters_three_layer_cnn_model)

    return model


def get_letters_CNN_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.letters_CNN_tf))

    return model


def get_digits_cnn_dg_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.digits_CNN_DataGen_tf))

    return model


def get_letters_cnn_dg_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.letters_CNN_DataGen_tf))

    return model


def get_digits_cnn_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.digits_CNN_tf))

    return model


def get_eng_letters_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.eng_letters_tf))

    return model


def get_eng_uppercase_letters_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.eng_uppercase_letters_tf))

    return model


def get_eng_uppercase_letters64_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.eng_uppercase_letters64_tf))

    return model


def get_eng_uppercase_letters64_v2_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.eng_uppercase_letters64_tf))

    return model


def get_touching_dgts_L_classfier_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.touching_dgts_L_classifier_tf_model))

    return model


def get_touching_dgts_C2_classfier_tf_model():
    model = load_model(os.path.join(config.MODELS_DIR, config.touching_dgts_C2_classifier_tf_model))

    return model
