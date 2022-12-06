from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# keras
import tensorflow as tf
# from keras import load_img
from keras.models import load_model
from keras.preprocessing import image


# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './model/model_f.h5'

# Load your trained model
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')

def get_key(val):
    if(val == 0):
      return "Infected"
    else:
      return "Not Infected"


def model_predict(img_path, model):
    img = tf.keras.utils.load_img(
    img_path,
    target_size=(224, 224)
    ,interpolation='nearest'
    ,keep_aspect_ratio=False)

    x = np.array(img)
    x = x/255.0
    x = x.reshape(1,224,224,3)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('./index.html')


@app.route('/aboutus', methods=['GET'])
def aboutus():
    # about us page
    return render_template('./about.html')

@app.route('/contactus', methods=['GET'])
def contactus():
    # contactus page
    return render_template('./contact.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)


        # j = preds.max()

        # l={"infected":preds[0][0],"not infected":preds[0][1]}

        # temp = preds[0][0]
        y_class = ((preds > 0.5)+0).ravel()

        ans = get_key(y_class)

        
        return ans
        
      

if __name__ == '__main__':
    app.run(debug=True)

