# -*- coding: utf-8 -*-
"""
    Flaskr
    ~~~~~~

    A microblog example application written as Flask tutorial with
    Flask and sqlite3.

    :copyright: (c) 2015 by Armin Ronacher.
    :license: BSD, see LICENSE for more details.
"""

import os
from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
import pandas as pd
import numpy as np
import pickle
import patsy


#export FLASK_APP=roo.py
#export FLASK_DEBUG=1

app = Flask(__name__)

# Load default config and override config from an environment variable
app.config.update(dict(
    #DATABASE=os.path.join(app.root_path, 'roo.db'),
    DEBUG=True,
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))
app.config.from_envvar('ROO_SETTINGS', silent=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    pass

@app.route('/model_predict', methods = ['POST'])
def model_predict():
    model =  pickle.load(
        open('/home/matt/kangaroo_parliament/roo/roo/models/model.p', 'rb'))
    prec = float(request.form['PREC'])
    educ = float(request.form['EDUC'])
    nonw = float(request.form['NONW'])
    xnew = pd.DataFrame({
            'PREC': np.array([prec]), 
            'EDUC': np.array([educ]), 
            'NONW': np.array([nonw])
    })
    formula =  'PREC + np.power(PREC,2) + EDUC + '
    formula += 'np.power(EDUC,2) + NONW + np.power(PREC,2)'
    xnew_trans = patsy.dmatrix(formula, data = xnew)
    pred = model.predict(xnew_trans, transform = False)
    entries = list(pred)
    return render_template('layout.html', entries=entries)