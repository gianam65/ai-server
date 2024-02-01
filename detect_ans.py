import tensorflow as tf
import imutils
import numpy as np
import cv2
from math import ceil
from collections import defaultdict
import matplotlib.pyplot as plt
# import subprocess
import math

class answer:
    def get_x_ver1(self,s):
        s = cv2.boundingRect(s)
        return s[0] * s[1]

    def get_x(self,s):
          return s[1][0]
    
    def pre_processing_img(self,img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        img_canny = cv2.Canny(blurred, 100, 200)

        cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts,gray_img

    @staticmethod
    def get_warped_image(dtd_path):
        ref = cv2.imread('./test_input/default_angle.png')
        
        rh, rw = ref.shape[:2]

        dtd = cv2.imread(dtd_path)
        dh, dw = dtd.shape[:2]

        cref = ref[56:rh-7, 5:rw-7]
        crh, crw = cref.shape[:2]

        cdtd = dtd[0:dh-5, 0:dw]
        cdh, cdw = cdtd.shape[:2]

        lower = (0, 0, 0)
        upper = (175, 175, 175)
        tref = cv2.inRange(cref, lower, upper)
        lower = (0, 0, 0)
        upper = (150, 150, 150)
        tdtd = cv2.inRange(cdtd, lower, upper)

        contours = cv2.findContours(tref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        ref_cntr = max(contours, key=cv2.contourArea)
        xr, yr, wr, hr = cv2.boundingRect(ref_cntr)

        rtl = (xr, yr)
        rtr = (xr + wr, yr)
        rbr = (xr + wr, yr + hr)
        rbl = (xr, yr + hr)

        ref_pts = np.float32([rtl, rtr, rbr, rbl])

        contours = cv2.findContours(tdtd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        dtd_cntr = max(contours, key=cv2.contourArea)
        rotrect = cv2.minAreaRect(dtd_cntr)
        (center), (width, height), angle = rotrect
        box = cv2.boxPoints(rotrect)
        boxpts = np.intp(box)

        (cx, cy) = center
        sort_info = []
        for pt in box:
            [px, py] = pt
            pxc = px - cx
            pyc = py - cy
            ang = (180 / math.pi) * math.atan2(pyc, pxc)
            sort_info.append([px, py, ang])

        def takeThird(elem):
            return elem[2]

        sort_info.sort(key=takeThird, reverse=False)

        x_info = []
        y_info = []
        for i in range(0, 4):
            [x, y, ang] = sort_info[i]
            x_info.append(x)
            y_info.append(y)

        dtl = (x_info[0], y_info[0])
        dtr = (x_info[1], y_info[1])
        dbr = (x_info[2], y_info[2])
        dbl = (x_info[3], y_info[3])

        dtd_pts = np.float32([dtl, dtr, dbr, dbl])
        matrix = cv2.getPerspectiveTransform(dtd_pts, ref_pts)
        dtd_warped = cv2.warpPerspective(cdtd, matrix, (crw, crh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        return dtd_warped

    def crop_image(self,img, mark_by_camera):
        cnts,gray_img = self.pre_processing_img(img)
        ans_blocks = []
        x_old, y_old, w_old, h_old = 0, 0, 0, 0

        if len(cnts) > 0:
            cnts = sorted(cnts, key=self.get_x_ver1)
            for i, c in enumerate(cnts):
                x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
                if w_curr * h_curr > 100000 and w_curr < h_curr:
                    check_xy_min = x_curr * y_curr - x_old * y_old
                    check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                    if len(ans_blocks) == 0:
                        not_rotated_img = gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr]
                        if mark_by_camera:
                            ans_block_img_path = "./temp_ans_block_img.jpg"  
                            cv2.imwrite(ans_block_img_path, not_rotated_img)
                            dtd_warped = self.get_warped_image(ans_block_img_path)
                            ans_blocks.append(
                                (dtd_warped, [x_curr, y_curr, w_curr, h_curr]))
                        else:
                            ans_blocks.append(
                                (not_rotated_img, [x_curr, y_curr, w_curr, h_curr]))

                        x_old,y_old,w_old,h_old = x_curr,y_curr,w_curr,h_curr

                    elif check_xy_min > 20000 and check_xy_max > 20000:
                        not_rotated_img = gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr]
                        if mark_by_camera:
                            ans_block_img_path = "./temp_ans_block_img.jpg"
                            cv2.imwrite(ans_block_img_path, not_rotated_img)
                            dtd_warped = self.get_warped_image(ans_block_img_path)
                            ans_blocks.append(
                                (dtd_warped, [x_curr, y_curr, w_curr, h_curr]))
                        else:
                            ans_blocks.append(
                                (not_rotated_img, [x_curr, y_curr, w_curr, h_curr]))
                        x_old,y_old,w_old,h_old = x_curr,y_curr,w_curr,h_curr

            sorted_ans_blocks = sorted(ans_blocks, key=self.get_x)
            return sorted_ans_blocks

    def divide_ans_blocks(self,ans_blocks):
        list_answers = []
        for ans_block in ans_blocks:
            ans_block_img = np.array(ans_block[0])
            offset1 = ceil(ans_block_img.shape[0] / 6)
            for i in range(6):
                    box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
                    height_box = box_img.shape[0]
                    box_img = box_img[14:height_box-14, :]
                    offset2 = ceil(box_img.shape[0] / 5)
                    for j in range(5):
                        list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])
        return list_answers
    
    def list_ans(self,list_answers, marked_by_camera):
        list_choices = []
        for answer_img in list_answers:
            start = 40
            offset = 46
            for i in range(4):
                bubble_choice = answer_img[:,start + i * offset:start + (i + 1) * offset]
                if marked_by_camera == True:
                    bubble_choice = cv2.cvtColor(bubble_choice, cv2.COLOR_BGR2GRAY) 
                
                bubble_choice = cv2.adaptiveThreshold(bubble_choice, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
            cv2.THRESH_BINARY,11,2)
                bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
                bubble_choice = bubble_choice.reshape((28, 28, 1))
                list_choices.append(bubble_choice)
                
        return list_choices
    
    def map_answer(self,idx):
        if idx % 4 == 0:
            answer_circle = "A"
        elif idx % 4 == 1:
            answer_circle = "B"
        elif idx % 4 == 2:
            answer_circle = "C"
        else:
            answer_circle = "D"
        return answer_circle
    
    def get_answers(self,list_answers,model):
        results = defaultdict(list)
        list_answers = np.array(list_answers)
        scores = model.predict_on_batch(list_answers / 255.0)
        for idx, score in enumerate(scores):
            question = idx // 4
            if score[1] > 0.9:
                chosed_answer = self.map_answer(idx)
                results[question + 1].append(chosed_answer)

        return results

def get_final_answer(img):
    img = cv2.imread(img)
    img = cv2.resize(img,(1100,1500))
    model  = tf.keras.models.load_model('weight.h5')
    crop =  answer()
    ans_blocks = crop.crop_image(img)
    list_answer = crop.divide_ans_blocks(ans_blocks)
    list_answer = crop.list_ans(list_answer)
    result = crop.get_answers(list_answer,model)
    return result

# def main():
#     # img = cv2.imread('./test_input/input_1.jpeg')
#     img = cv2.imread('corrected.png')
#     img = cv2.resize(img,(1100,1500))
#     model = tf.keras.models.load_model('weight.h5')
#     crop =  answer()
#     ans_blocks = crop.crop_image(img, True)
#     list_answer = crop.divide_ans_blocks(ans_blocks)
#     list_answer = crop.list_ans(list_answer, True)
#     result = crop.get_answers(list_answer,model)
#     print("result", result)
#     return result

# main()




