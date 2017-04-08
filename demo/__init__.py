# -*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import request
import flask
import numpy as np

import cvae
import tensorflow as tf

from sklearn.preprocessing import binarize

nb_classes = 72
labels = u"あいうえおか平がきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもやゆよら開りるれろわん"
app = Flask(__name__, static_url_path='/static')
model = cvae.VAE()
model.load('./cvae.ckpt')

y_label = []
for i in range(nb_classes):
        a = [0] * nb_classes
        a[i] = 1
        y_label.append(a)

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
    y_ =[0] * nb_classes
    y_[idx] = 1
    y = np.array([y_])
    x_ = model.reconstruct(x, y)
    z = model.infer(x, y)
    result = (255 - x_*5 * 255).astype(np.int32).tolist()

    result += hoge(np.tile(z, [nb_classes, 1]), np.array(y_label))
    print len(result)

    return flask.jsonify({'pred':labels[idx], 'result':result})

def hoge(z, y):
    x_ = model.generate(z, y)
    return (255 - x_ *5* 255).astype(np.int32).tolist()


if __name__ == '__main__':
    app.run()
