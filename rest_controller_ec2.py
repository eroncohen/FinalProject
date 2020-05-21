# Serve model as a flask application

import numpy as np
from flask import Flask, request
import cv2
import tensorflow as tf
from image_analyzer import FaceCropper
from tensorflow.python.keras.backend import set_session
sess = None
graph = None
detector = None
app = Flask(__name__)


def initialize():
    global sess
    global graph
    global detector
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    set_session(sess)
    graph = tf.compat.v1.get_default_graph()
    detector = FaceCropper()


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        r = request
        # convert string of image data to uint8
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)
        with sess.as_default():
            with graph.as_default():
                prediction = detector.generate(img)
        return str(prediction[0][0])


if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000)
