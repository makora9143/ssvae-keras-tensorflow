# -*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import request
import flask
import numpy as np

import cvae
import tensorflow as tf

from sklearn.preprocessing import binarize

nb_classes = 10
hiragana_labels = u"あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよわん"
mnist_labels= "0123456789"
labels = mnist_labels
app = Flask(__name__, static_url_path='/static')

model = cvae.VAE()
model.load('./mnist.ckpt')



@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/api/change', methods=['POST'])
def change_mode():
    mode = request.json['mode']
    print mode
    if mode == 'mnist':
        model.close()
        model.build_model("mnist")
        model.load('./mnist.ckpt')
        return flask.jsonify({'msg': 'success'})
    elif mode == 'hiragana':
        model.close()
        model.build_model("hiragana")
        model.load('./hiragana.ckpt')
        nb_classes = 70
        return flask.jsonify({'msg': 'success'})
    else:
        return flask.jsonify({'msg': 'fail'})

@app.route('/api/recon', methods=['POST'])
def reconstruct():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    x = (255. - np.array([request.json]).astype(np.float32)) / 255.
    p_r = model.predict(x)
    idx = np.argmax(p_r)
    y_ =[0] * p_r.shape[1]
    y_[idx] = 1
    y = np.array([y_])
    x_ = model.reconstruct(x, y)
    z = model.infer(x, y)
    y_label = []
    for i in range(p_r.shape[1]):
            a = [0] * p_r.shape[1]
            a[i] = 1
            y_label.append(a)
    result = (255 - x_ * 255).astype(np.int32).tolist()
    result += hoge(np.tile(z, [p_r.shape[1], 1]), np.array(y_label))

    return flask.jsonify({'pred':idx, 'result':result})

def hoge(z, y):
    x_ = model.generate(z, y)
    return (255 - x_ * 255).astype(np.int32).tolist()


if __name__ == '__main__':
    app.run()
