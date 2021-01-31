# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:49:10 2021

@author: saketh
"""

from flask import Flask,request,render_template
import os
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
model = pickle.load(open('PAE_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict',methods=["POST","GET"])

def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    print(features_value)
    
    features_name = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'department', 'salary']
    
    scaler=pickle.load(open("scaler.pkl","rb"))
    X_test_scaled=scaler.transform(features_value)
    prediction = model.predict(X_test_scaled)
    output=prediction[0]    
   
    return render_template('result.html', prediction_text=output)
if __name__=="__main__":
     
     app.run(debug=True)
     