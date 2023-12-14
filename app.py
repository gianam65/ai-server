from flask import Flask, request, jsonify
from process_img import crop_image, process_ans_blocks, process_list_ans, get_answers, get_sbd, get_md, calculate_score
from main import CNN_Model
import cv2
import numpy as np
from flask_cors import CORS
import json 
from json import JSONEncoder
from detect_ans import answer
import tensorflow as tf

app = Flask(__name__)
CORS(app)

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img,(1100,1500))

    default_result = json.loads(request.form['default_result'])
    # list_ans_boxes = crop_image(img)
    # list_ans = process_ans_blocks(list_ans_boxes)
    # list_ans = process_list_ans(list_ans)
    # answers = get_answers(list_ans)

    model  = tf.keras.models.load_model('weight.h5')
    crop = answer()  
    ans_blocks = crop.crop_image(img)
    list_ans = crop.divide_ans_blocks(ans_blocks)
    list_ans = crop.list_ans(list_ans)
    answers = crop.get_answers(list_ans, model)
    need_re_mark = any(len(values) > 1 for values in answers.values())

    score = calculate_score(answers, {i + 1: [default_result[i]] for i in range(len(default_result))})
    sbd = get_sbd(img)
    md = get_md(img)

    response = {
        'answers': answers,
        'score': score,
        'sbd': sbd,
        'md': md,
        "need_re_mark": need_re_mark
    }

    return json.dumps(response, cls=NumpyEncoder)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
