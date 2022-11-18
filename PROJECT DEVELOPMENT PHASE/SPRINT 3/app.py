 #Flask-framework to run/serve application
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from flask_cors import CORS
import flask
import pandas as pd
from flask import request, render_template

app = flask.Flask(__name__, static_url_path='')

@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/web', methods=['GET'])
def recognize():
    return render_template('web.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28,28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        model = load_model('models/mnistCNN.h5')
        predicted = model.predict(im2arr)
        predicted = np.argmax(predicted, axis=1)
    return render_template('web.html', predicted=str(predicted[0]))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

