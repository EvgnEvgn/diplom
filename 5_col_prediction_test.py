import cv2
import numpy as np
from neural_nets.model_utils import *
from extracting_data import segment_touching_digits
import os
import matplotlib.pyplot as plt

dir_path = r"Output/ExtractedColumns/5_col"
img_paths = os.listdir(dir_path)
Length_model = get_touching_dgts_L_classfier_tf_model()
Touching2Digit_model = get_touching_dgts_C2_classfier_tf_model()

for img_path in img_paths:
    result_path = os.path.join(dir_path, img_path)
    img = cv2.imread(result_path)
    segmented_img = segment_touching_digits(img)
    if segmented_img is not None:
        data = np.reshape(segmented_img, (1, 64, 64, 1))
        data = np.divide(data, 255).astype("float32")
        probs = Length_model.predict([data])
        predicted = Length_model.predict_classes([data])
        print(probs)
        print(predicted)

        predicted_probs = Touching2Digit_model.predict([data])[0]

        predicted_prob_indxs = np.argsort(predicted_probs)[::-1]
        top3_predicted_class = predicted_prob_indxs[0:3]
        print(top3_predicted_class)
        print(predicted_probs[top3_predicted_class])
        predicted_2touching_class = Touching2Digit_model.predict_classes([data])

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Orig')
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_img)
        plt.title('Segmented: symb count = {0}, class = {1}'.format(predicted[0] + 1, predicted_2touching_class))
        plt.show()
    else:
        plt.imshow(img)
        plt.title('Orig')
        plt.show()
