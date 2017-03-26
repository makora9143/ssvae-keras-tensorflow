# -*- coding: utf-8 -*-

from flask import Flask
from flask import render_template
from flask import request
import flask
import numpy as np

import vae

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/api/recon', methods=['POST'])
def reconstruct():
    if request.headers['Content-Type'] != 'application/json':
        print(request.headers['Content-Type'])
        return flask.jsonify(res='error'), 400

    x = np.array([request.json])
    print x.shape

    model = vae.M2VAE()
    model.load('/home/makora/notebooks/demo.ckpt')

    #y = model.predict(x)
    #z = model.inference(x, y)
    #result = [model.reconstruct(x, y)]
    #result += [model.generate(z, i) for i in range(10)]

    result = []
    return flask.jsonify(result)



if __name__ == '__main__':
    app.run()
