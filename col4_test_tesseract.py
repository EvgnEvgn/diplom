import pytesseract
import matplotlib.pyplot as plt
import string
from extracting_data import get_letter_images_from_col5
import cv2

test_labels = ["c", "a", "d", "c", "b", "c", "a", "e", "", "b",
               "b", "d", "b", "e", "c", "b", "a", "a", "b", "b",
               "b", "", "a", "a", "b", "a", "a", "a", "", "b",
               "b", "", "", "a", "b", "", "", "a", "a", "b",
               "", "c", "b", "b", "b", "b", "", "d", "b", "c",
               "c", "b", "", "", "a", "b", "a", "d", "a", "",
               "b", "b", "", "", "", "a", "a", "", "a", "a"]

alphabet = dict((idx, key) for idx, key in enumerate(string.ascii_lowercase))
bad_result = []
img = cv2.imread("Output/ExtractedColumns/1_col/0.jpg")
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img_bin) = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY_INV)
letters, paths = get_letter_images_from_col5()
predicted = pytesseract.image_to_string(img_bin, lang='eng')
print(predicted)
plt.imshow(img_bin, cmap='gray')
plt.show()
