import cv2
from extracting_data import *
from neural_nets.data_utils import *
import pickle
from align_images import *

# def deskew_v2(img):
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         # no deskewing needed.
#         return img.copy()
#     # Calculate skew based on central momemts.
#     skew = m['mu11'] / m['mu02']
#     # Calculate affine transform to correct skewness.
#     M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
#     # Apply affine transform
#     img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
#     return img


test_labels = ["c", "a", "d", "c", "b", "c", "a", "e", "", "b",
               "b", "d", "b", "e", "c", "b", "a", "a", "b", "b",
               "b", "", "a", "a", "b", "a", "a", "a", "", "b",
               "b", "", "", "a", "b", "", "", "a", "a", "b",
               "", "c", "b", "b", "b", "b", "", "d", "b", "c",
               "c", "b", "", "", "a", "b", "a", "d", "a", "",
               "b", "b", "", "", "", "a", "a", "", "a", "a"]

alphabet = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "X"}
bad_result = []


# letters, paths = get_letters_img_for_test()
#
# letters_model = get_eng_letters_tf_model()
# predicted = letters_model.predict_classes([letters])
#
# for idx, l in enumerate(letters):
#     img = cv2.imread(os.path.join("Output/eng_letters_test_data", paths[idx]))
#     cv2.imshow(paths[idx], img)
#     cv2.imshow(alphabet[predicted[idx]], letters[idx])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def predict_eng_letters_imgs():
    path = r"Output/eng_letters_with_issues"
    letters_model = get_eng_uppercase_letters64_v2_tf_model()

    listdir = os.listdir(path)
    paths = np.array([])
    idx = 0
    for img_path in listdir:
        full_path = os.path.join(path, img_path)
        if os.path.isfile(full_path):
            img = cv2.imread(full_path)
            letters = segment_eng_letters(img)
            sub_indx = 0
            for l in letters:
                cv2.imwrite(os.path.join("Output/EngLettersCropResult",
                                         "cropped_letter_" + str(idx) + "_" + str(sub_indx) + ".jpg"),
                                         l)
                sub_indx += 1
                data = np.reshape(l, (1, 64, 64, 1))
                data = np.divide(data, 255).astype("float64")
                probs = letters_model.predict([data])
                predicted = letters_model.predict_classes([data])
                plt.subplot(1, 2, 1)
                plt.title("Orig - " + img_path)
                plt.imshow(img)
                plt.subplot(1, 2, 2)
                plt.imshow(l)
                plt.title("Predicted - " + alphabet[predicted[0]])
                plt.show()
                print(probs)
                print(predicted)

            idx += 1


# img = cv2.imread("Output/eng_letters_with_issues/48.jpg")
#
# grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# rotated = deskew_v2(thresh)
# cv2.imwrite(os.path.join("Output/CroppedImages", "rotated.jpg"), rotated)
# sheared_img = unshear(rotated)
# cv2.imwrite(os.path.join("Output/CroppedImages", "sheared_img.jpg"), sheared_img)
#
# plt.imshow(sheared_img)
# plt.show()
# letters = segment_letters_v2(sheared_img)

# img = cv2.imread("Output/AligningResult/a_letter_6.jpg", 0)
# _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# #img = cv2.erode(img, np.ones((2, 2)), iterations=1)
# #img = cv2.dilate(img, np.ones((2, 2)), iterations=2)
#
# img = cv2.resize(img, ((28, 28)), interpolation=cv2.INTER_AREA)
# # img = np.pad(img, ((4, 4), (4, 4)))
#
# plt.imshow(img)
# plt.show()
# img = np.reshape(img, (28, 28, 1))
# a = np.zeros((1, 28, 28, 1))
# a[0] = img
# a = np.divide(a, 255).astype("float64")
# letters_model = get_eng_letters_tf_model()
# probs = letters_model.predict([a])
# predicted = letters_model.predict_classes([a])
# print(probs)
# print(predicted)

predict_eng_letters_imgs()
