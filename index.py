import time
from datetime import timedelta
from functools import wraps
import uuid
from flask import Flask, render_template, Response, jsonify, session, request, redirect, url_for, g
from cv2 import cv2 as cv
from VideoCapture import VideoCapture
from utilities import predict_sign
from skimage.metrics import structural_similarity


app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "12345@#$%"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=2)


def token_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'tokenized' in session:
            print("Success")
            return f(*args, *kwargs)
        else:
            print("Your session expired")
            return "Your session expired"
    return wrap


# Runs camera using VideoCapture class
def get_camera(camera):
    bg_start = None
    diff_start = None
    while True:
        # Creates camera object
        frame, video_frame = camera.get_video_capture()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + video_frame + b'\r\n\r\n')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    screen = cv.VideoCapture(0)
    ret, frame = screen.read()
    pred = predict_sign(frame, 0)
    response = jsonify({'message': pred})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    session.permanent = True
    session.modified = True
    new_token = str(uuid.uuid4())
    session['tokenized'] = new_token
    response = jsonify({'message': 'New session created'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/check_session')
def check_session():
    if 'tokenized' in session:
        return jsonify({'Session': 'true'})
    else:
        return jsonify({'Session': 'false'})


@token_required
@app.route('/video_page/<resolution>/', methods=['GET', 'POST'])
def video_page(resolution):
    if session.get('tokenized'):
        print(session['tokenized'])
    if 'resolution' in session:
        if session['resolution'] != resolution:
            session['resolution'] = resolution
    return Response(get_camera(VideoCapture(resolution)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, threaded=True, use_reloader=False)
