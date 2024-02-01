from flask import Flask, request, jsonify
from process_img import get_sbd, get_md, calculate_score
from main import CNN_Model
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from flask_cors import CORS
import json 
from json import JSONEncoder
from detect_ans import answer
import tensorflow as tf
import subprocess
# from flask_migrate import Migrate
import cloudinary
from cloudinary.uploader import upload
import os

cloudinary.config(
    cloud_name="dtxvbyskh",
    api_key="531672898283468",
    api_secret="pwpvcEJMQIzZm1cTmam1ub6Ms4g"
)

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
    classes = db.Column(db.String(50))
    answers = db.relationship('AnswerItem', backref='answer', lazy=True)
    correct_answer = db.Column(db.Text)
    image_url = db.Column(db.String(255))

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

def save_image_to_external_service():
    try:
        response = upload(
            './test_input/user_uploaded_image.jpeg',
            folder="images"
        )

        image_url = response.get("secure_url")

        return image_url
    except Exception as e:
        print(f"Error uploading image to Cloudinary: {e}")

    return ""

@app.route('/process_image', methods=['POST'])
def process_image():
    mark_by_camera = request.form.get('mark_by_camera')
    mark_by_camera = bool(mark_by_camera) if mark_by_camera else False
    image_file = request.files['image']
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img,(1100,1500))

    file_extension = os.path.splitext(image_file.filename)[1].lower()
    default_result = json.loads(request.form['default_result'])
    saved_img_path = "./test_input/user_uploaded_image.jpeg"

    if mark_by_camera:
        cv2.imwrite(saved_img_path, img)
        script_path = "pyimgscan.py"
        image_path = saved_img_path
        command = ["python3", script_path, "-i", image_path]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Command output:", result.stdout)
        except subprocess.CalledProcessError as e:
            return jsonify({'success': False, 'error_message': 'Ảnh đầu vào quá mờ hoặc không đủ độ sáng'})

        img = cv2.imread("corrected.png")

    cv2.imwrite(saved_img_path, img)
    model  = tf.keras.models.load_model('weight_test.h5')
    crop = answer()  
    ans_blocks = crop.crop_image(img, mark_by_camera or file_extension == ".png")
    list_ans = crop.divide_ans_blocks(ans_blocks)
    list_ans = crop.list_ans(list_ans, mark_by_camera or file_extension == ".png")
    answers = crop.get_answers(list_ans, model)
    need_re_mark = any(len(values) > 1 for values in answers.values())

    score = calculate_score(answers, {i + 1: [default_result[i]] for i in range(len(default_result))})
    sbd = get_sbd(img)
    md = get_md(img)
    image_url = save_image_to_external_service()

    response = {
        'success': True,
        'answers': answers,
        'score': score,
        'sbd': ''.join(map(str, sbd)),
        'md': ''.join(map(str, md)),
        "need_re_mark": need_re_mark,
        'classes': request.form['classes'],
        'correct_answer': default_result,
        'image_url': image_url,
        'correct_answer': request.form['default_result'],
    }
    with app.app_context():
        new_answer_id = save_to_db(response)

    response['id'] = new_answer_id  

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
            'classes': answer.classes,
            'image_url': answer.image_url,
            'correct_answer': answer.correct_answer,
            'answers': [{'question_number': item.question_number, 'answer_options': json.loads(item.answer_options)} for item in answer.answers]
        }
        answer_list.append(answer_dict)

    return jsonify({'answers': answer_list})

@app.route('/get_answer/<int:answer_id>', methods=['GET'])
def get_answer_by_id(answer_id):
    answer = Answer.query.get(answer_id)

    if answer is not None:
        response = {
            'id': answer.id,
            'score': answer.score,
            'sbd': answer.sbd,
            'md': answer.md,
            'need_re_mark': answer.need_re_mark,
            'image_url': answer.image_url,
            'classes': answer.classes,
            'correct_answer': answer.correct_answer,
            'answers': [{'question_number': item.question_number, 'answer_options': json.loads(item.answer_options)} for item in answer.answers]
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'Answer not found'}), 404


def save_to_db(response):
    new_answer = Answer(
        score=response['score'],
        sbd=''.join(map(str, response['sbd'])),  
        md=''.join(map(str, response['md'])), 
        need_re_mark=response['need_re_mark'],
        classes=response['classes'],
        correct_answer=json.dumps(response['correct_answer']),
        image_url= response['image_url']
    )

    for question_number, answer_options in response['answers'].items():
        answer_item = AnswerItem(
            question_number=question_number,
            answer_options=json.dumps(answer_options, cls=NumpyEncoder)
        )
        new_answer.answers.append(answer_item)

    db.session.add(new_answer)
    db.session.commit()
    return new_answer.id

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, ssl_context=('./cert.pem', './key.pem'))

# migrate = Migrate(app, db)