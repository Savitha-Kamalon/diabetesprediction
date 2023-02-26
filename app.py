import pickle
from flask import Flask,render_template,app, request,jsonify,url_for
import numpy as np
import pandas as pd
app=Flask(__name__)
rfmodel = pickle.load(open('rfmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')


if __name__=="__main__":
    app.run(debug=True) 