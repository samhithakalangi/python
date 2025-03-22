#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask,request,jsonify
import joblib
import numpy as np

app=Flask(__name__)

# load the tarined model
model=joblib.load("model.pkl")

@app.route('/')
def home():
    return "Welcome to the ML model api!"

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json()
    features=np.array(data['features']).reshape(1,-1)
    prediction=model.predict(features)
    return jsonify({'prediction':prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

