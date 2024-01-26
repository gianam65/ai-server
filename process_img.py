import numpy as np
import cv2
import tensorflow as tf
import crop_image as ci

def calculate_score(actual_result, default_result):
    correct_count = 0
    total_questions = len(default_result)

    for question, actual_answers in actual_result.items():
        default_answers = default_result[question] if question in default_result else []

        actual_answers = list(map(str, actual_answers))
        default_answers = list(map(str, default_answers))

        if actual_answers == default_answers:
            correct_count += 1

    score = (correct_count / total_questions) * 100
    return score

def get_sbd(img):
    crop = ci.crop_image()
    a = crop.crop_image_sbd(img)
    b = crop.split_blocks_sbd(a)
    
    model_digit = tf.keras.models.load_model('weight_digit.h5')

    def predict(img, model):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret, im_th = cv2.threshold(img_gray, 179, 255, cv2.THRESH_BINARY_INV)
        roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi / 255.0
        y = model.predict(roi.reshape(-1, 28, 28, 1))
        return np.argmax(y)

    return [predict(block, model_digit) for block in b]

def get_md(img):
    crop = ci.crop_image()
    a1 = crop.crop_image_md(img)
    b1 = crop.split_blocks_md(a1)
    
    model_digit = tf.keras.models.load_model('weight_digit.h5')

    def predict(img, model):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        ret, im_th = cv2.threshold(img_gray, 179, 255, cv2.THRESH_BINARY_INV)
        roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi / 255.0
        y = model.predict(roi.reshape(-1, 28, 28, 1))
        return np.argmax(y)

    return [predict(block, model_digit) for block in b1]


# if __name__ == '__main__':
#     img = cv2.imread('./test_input/asd.png')
#     # list_ans_boxes = crop_image(img)
#     # list_ans = process_ans_blocks(list_ans_boxes)
#     # list_ans = process_list_ans(list_ans)
#     # answers = get_answers(list_ans)

#     results_sbd_list = get_sbd(img)
#     print("results_sbd_list", results_sbd_list)
#     results_md_list = get_md(img)
