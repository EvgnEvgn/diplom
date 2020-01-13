import cv2
import pytesseract
from extracting_data import *

true_words = [
    "Костенко",
    "Даниил",
    "Юрьевич",
    "Куликова",
    "Анастасия",
    "Вячеславовна",
    "Малярова",
    "Светлана",
    "Андреевна",
    "Мосина",
    "Дана",
    "Юрьевна",
    "Мысенко",
    "Данил",
    "Александрович",
    "Нечунаев",
    "Алексей",
    "Геннадиевич",
    "Шелухин",
    "Данил",
    "Иванович",
    "Плишак",
    "Павел",
    "Леонидович",
    "Притыкин",
    "Семён",
    "Аркадьевич",
    "Радченко",
    "Тарас",
    "Викторович",
    "Синий",
    "Александр",
    "Александрович",
    "Хлистунов",
    "Даниил",
    "Александрович",
    "Ящук",
    "Максим",
    "Александрович",
    "Костенко",
    "Даниил",
    "Юрьевич",
    "Куликова",
    "Анастасия",
    "Вячеславовна"
]
test_count = 45
idx = 0
counter = 0
prediction_result = 0
col_path = r"Output/ExtractedColumns/1_col"
bad_result = []
paths = ["{0}.jpg".format(i) for i in range(0,15)]
for path in paths:
    if idx >= 15:
        break

    full_path = os.path.join(col_path, path)
    if os.path.isfile(full_path):
        img = cv2.imread(full_path)
        img_words = segment_characters_from_col2(img)
        for img_word in img_words:
            predicted_word = pytesseract.image_to_string(img_word, lang='rus')
            print(predicted_word)
            if predicted_word == true_words[counter]:
                prediction_result += 1
            else:
                bad_result.append(counter)
            counter += 1
        idx += 1

print(counter)
print(prediction_result)
print(prediction_result/test_count)
print(bad_result)