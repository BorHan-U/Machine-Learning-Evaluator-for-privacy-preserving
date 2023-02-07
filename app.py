import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
modelAccuracy = pickle.load(open('modela.pkl', 'rb'))
modelRecall = pickle.load(open('modelr.pkl', 'rb'))
modelPrecision = pickle.load(open('modelp.pkl', 'rb'))
modelF1_Score = pickle.load(open('modelf.pkl', 'rb'))
modelLoss = pickle.load(open('modell.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    modelAccuracy
    modelRecall
    modelPrecision
    modelF1_Score
    modelLoss
    # return render_template('index.html', prediction_text='Accuracy' (model))
    return render_template('index.html', accuracy_score='Accuracy = {}'.format(modelAccuracy),
                           recall_score='Recall = {}'.format(modelRecall), precision_score='Precision = {}'.format(modelPrecision), f1_score='f1_score = {}'.format(modelF1_Score), barier_score='Barier_score = {}'.format(modelLoss))


if __name__ == "__main__":
    app.run(debug=True)
