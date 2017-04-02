# -*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import request
import flask
import numpy as np

import cvae
import tensorflow as tf

from sklearn.preprocessing import binarize

app = Flask(__name__, static_url_path='/static')
model = cvae.VAE()
model.load('./cvae.ckpt')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/api/recon', methods=['POST'])
def reconstruct():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    x = (255. - np.array([request.json]).astype(np.float32)) / 255.
    idx = np.argmax(model.predict(x))
    print idx
    y_ =[0] * 10
    y_[idx] = 1
    y = np.array([y_])
    x_ = model.reconstruct(x, y)
    z = model.infer(x, y)
    result = (255 - x_ * 255).astype(np.int32).tolist()

    y_label = []
    for i in range(10):
            a = [0] * 10
            a[i] = 1
            y_label.append(a)
    result += hoge(np.tile(z, [10, 1]), np.array(y_label))

    return flask.jsonify({'pred':idx, 'result':result})

def hoge(z, y):
    x_ = model.generate(z, y)
    return (255 - x_ * 255).astype(np.int32).tolist()


if __name__ == '__main__':
    app.run()
