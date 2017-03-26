# -*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import request
import flask
import numpy as np

import vae
import tensorflow as tf

app = Flask(__name__, static_url_path='/static')
model = vae.M2VAE()
#model.load('./demo.ckpt')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/api/recon', methods=['POST'])
def reconstruct():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    x = np.array([request.json]).astype(np.float32) / 255.
    y = model.classify(x)
    z = model.infer(x, y)
    result = [model.generate(z, y).tolist()]

    y_lable = []
    for i in range(10):
            a = [0] * 10
            a[i] = 1
            y_lable.append(a)
    result += (model.generate(np.tile(z, [10, 1]), np.array(y_lable)) * 255.).tolist()

    return flask.jsonify(result)



if __name__ == '__main__':
    app.run()
