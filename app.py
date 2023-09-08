from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from keras_vggface import utils
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg16


app = Flask(__name__, template_folder='.')

def process_arr(arr, version = 2):
    img = cv2.resize(arr, (224, 224)).astype('float32')
    img = np.expand_dims(img, 0)
    img = img/255.0
    img = utils.preprocess_input(img, version = version)
    return img

model = tf.keras.models.load_model('./face_to_bmi.h5', compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

# Set the desired width and height for the streamed video
FRAME_WIDTH = 200
FRAME_HEIGHT = 480

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

@app.route('/')
def index():
    return "Real-Time Face Detection and Prediction"

def generate_frames():
    while True:
        ret, frame = video_capture.read()
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_image =  process_arr(face_image)
            bmi, sex = model.predict(face_image)
            cv2.putText(frame, 'BMI:{:3.1f} SEX:{:s}'.format(bmi[0, 0], 'M' if sex[0, 0] > 0.05 else 'F'),
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
@app.route('/main')
def other_page():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(ssl_context='adhoc', host='0.0.0.0', port=8000)