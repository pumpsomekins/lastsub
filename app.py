import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])

    return render_template('index.html', prediction_text='Crime should be type {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)