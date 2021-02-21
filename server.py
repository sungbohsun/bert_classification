#!/usr/bin/env python
import sys
import os
from os import path
sys.path.insert(0, path.join(path.dirname(__file__), '../src'))

from flask import Flask, render_template, url_for, request
from flask_assets import Environment, Bundle
from demo import Ans_setup

app = Flask(__name__,static_folder='static',template_folder='templates')
Model = Ans_setup(6,4,0.7)

assets = Environment(app)
assets.register('main',
                'main.css', 'font-awesome.min.css',
                output='cached.css', filters='cssmin')


#網頁執行/say_hello時，會導至index.html
@app.route('/', methods=['GET'])
def getdata():
    return render_template('index.html')

#index.html按下submit時，會取得前端傳來的username，並回傳"Hello, World! "+name
@app.route('/', methods=['POST'])
def submit():
    name = request.form.get('username')
    Qus,Ans = Model.Ans(name)
    
    return render_template('index.html', **locals())




if __name__ == '__main__':
    
    port = int(os.environ.get('PORT', 7967))
    app.debug = True
    app.run(threaded = True,host='0.0.0.0', port=port)