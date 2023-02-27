
from flask import Flask,render_template,app, request,jsonify,url_for
import numpy as np
import pandas as pd
import pickle
app=Flask(__name__)
rfmodel = pickle.load(open('rfmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = rfmodel.predict(new_data)
    print(output[0])
    return jsonify(float(output[0]))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input =scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    prediction = rfmodel.predict(final_input)[0]
    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred  
    return render_template("home.html",prediction_text = "{}".format(output))


if __name__=="__main__":
    app.run(debug=True) 