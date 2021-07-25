from flask.json import htmlsafe_dump
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]

    prediction = model.predict(int_features)

    output = prediction[0]


    return render_template('index.html', prediction_text='Statement is : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)