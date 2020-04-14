import cv2
#import pytesseract
from extracting_data import *
from neural_nets.data_utils import *
import pickle

test_labels = ["c", "a", "d", "c", "b", "c", "a", "e", "", "b",
               "b", "d", "b", "e", "c", "b", "a", "a", "b", "b",
               "b", "", "a", "a", "b", "a", "a", "a", "", "b",
               "b", "", "", "a", "b", "", "", "a", "a", "b",
               "", "c", "b", "b", "b", "b", "", "d", "b", "c",
               "c", "b", "", "", "a", "b", "a", "d", "a", "",
               "b", "b", "", "", "", "a", "a", "", "a", "a"]

alphabet = dict((idx, key) for idx, key in enumerate(string.ascii_lowercase))
bad_result = []

letters, paths = collect_letters_from_col5()

# x_train, y_train, x_val, y_val, x_test, y_test = get_emnist_letters_data_TF()
# letters_model = get_letters_cnn_dg_tf_model()
# predicted = letters_model.predict_classes([letters])

# with open("hand_letters_pred.dat", "wb") as file:
#   pickle.dump(predicted, file)

# print(predicted)
with open("hand_letters_pred.dat", "rb") as file:
    predicted = pickle.load(file)

skipped = 0
pure_symbol_count = 0
pred_accuracy = 0.0
predicted_count = 0
for idx, pred in enumerate(predicted):
    true_char = test_labels[idx]
    if true_char == "":
        skipped += 1
        continue

    pred_char = alphabet[pred]
    print("True character: {0} | Predicted character: {1}".format(true_char, pred_char))
    if pred_char == true_char:
        predicted_count += 1
    pure_symbol_count += 1

pred_accuracy = predicted_count / pure_symbol_count
print("Test count: ", pure_symbol_count)
print("Test accuracy: ", pred_accuracy)
