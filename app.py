from flask import Flask, request, jsonify
from process_img import crop_image, process_ans_blocks, process_list_ans, get_answers, get_sbd, get_md, calculate_score
from main import CNN_Model
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from flask_cors import CORS
import json 
from json import JSONEncoder
from detect_ans import answer
import tensorflow as tf

app = Flask(__name__)
CORS(app)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///answers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Float)
    sbd = db.Column(db.String(50))
    md = db.Column(db.String(50))
    need_re_mark = db.Column(db.Boolean)
    answers = db.relationship('AnswerItem', backref='answer', lazy=True)

class AnswerItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_number = db.Column(db.Integer)
    answer_options = db.Column(db.String(255))
    answer_id = db.Column(db.Integer, db.ForeignKey('answer.id'), nullable=False)

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def create_app():
    with app.app_context():
        db.create_all()

        model = tf.keras.models.load_model('weight_test.h5')

        return model

model = create_app()

@app.route('/')
def home():
    return 'TEST API'

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files['image']
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img,(1100,1500))

    default_result = json.loads(request.form['default_result'])

    model  = tf.keras.models.load_model('weight_test.h5')
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
        'sbd': ''.join(map(str, sbd)),
        'md': ''.join(map(str, md)),
        "need_re_mark": need_re_mark
    }
    with app.app_context():
        save_to_db(response)

    return json.dumps(response, cls=NumpyEncoder)

@app.route('/get_answers', methods=['GET'])
def get_answers():
    all_answers = Answer.query.all()
    answer_list = []

    for answer in all_answers:
        answer_dict = {
            'id': answer.id,
            'score': answer.score,
            'sbd': answer.sbd,
            'md': answer.md,
            'need_re_mark': answer.need_re_mark,
            'answers': [{'question_number': item.question_number, 'answer_options': json.loads(item.answer_options)} for item in answer.answers]
        }
        answer_list.append(answer_dict)

    return jsonify({'answers': answer_list})

def save_to_db(response):
    new_answer = Answer(
        score=response['score'],
        sbd=''.join(map(str, response['sbd'])),  
        md=''.join(map(str, response['md'])), 
        need_re_mark=response['need_re_mark']
    )

    for question_number, answer_options in response['answers'].items():
        answer_item = AnswerItem(
            question_number=question_number,
            answer_options=json.dumps(answer_options, cls=NumpyEncoder)
        )
        new_answer.answers.append(answer_item)

    db.session.add(new_answer)
    db.session.commit()

if __name__ == '__main__':
    app.run(port=8000, debug=True)
