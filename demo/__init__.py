# -*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import request
import flask
import numpy as np

import vae
import tensorflow as tf

from sklearn.preprocessing import binarize

app = Flask(__name__, static_url_path='/static')
model = vae.M2VAE()
model.load('../demo.ckpt')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/api/recon', methods=['POST'])
def reconstruct():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    x = (255. - np.array([request.json]).astype(np.float32)) / 255.
    y = model.classify(x)
    idx = np.argmax(y, axis=1)[0]
    y_ =[0] * 10
    y_[idx] = 1
    print idx
    y = np.array(y)
    z, _ = model.infer(x)
    x_ = binarize(model.generate(z, y), 0.2)
    result = (255 - x_ * 255).astype(np.int32).tolist()

    y_lable = []
    for i in range(10):
            a = [0] * 10
            a[i] = 1
            y_lable.append(a)
    result += hoge(np.tile(z, [10, 1]), np.array(y_lable))

    return flask.jsonify(result)

def hoge(z, y):
    x_ = binarize(model.generate(z, y), 0.2)
    return (255 - x_ * 255).astype(np.int32).tolist()


if __name__ == '__main__':
    app.run()
