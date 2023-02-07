from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename

# *** Flask configuration

# Define folder to save uploaded files to process further
# UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
UPLOAD_FOLDER = os.path.join(
    '/Users/borhan/Desktop/projectMadam/Deployment-flask-master')

# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__, template_folder='templates',
            static_folder='static/css')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'


@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')


@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']

        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)

        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename))

        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(
            app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('index_upload_and_show_data_page2.html')


@app.route('/show_result')
def showResult():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)


# Importing the libraries
    from sklearn.linear_model import LinearRegression
    import numpy as np
    import matplotlib.pyplot as plt
    # import pandas as pd
    import pickle

    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import brier_score_loss

    from sklearn.preprocessing import OrdinalEncoder
    from sklearn import metrics
    from sklearn.metrics import accuracy_score

    # Import Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier

    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    # flask demo ori model........
    # dataset = pd.read_csv('pima diabetes(email).csv')

    # flask demo ori model........
    # dataset = pd.read_csv('hiring.csv')

    # split dataset in features and target variable
    feature_cols = ['Pregnancies', 'Insulin', 'BMI',
                    'Age', 'Glucose', 'BloodPressure', 'Email', 'DiabetesPedigreeFunction']
    X = uploaded_df[feature_cols]  # Features
    y = uploaded_df.Outcome  # Target variable

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=1)  # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # print(y_pred)

    # print(accuracy)
    accuracy = round(accuracy_score(y_pred, y_test), 4)
    print(accuracy)

    # print('Recall: %.3f' % recall_score(y_test, y_pred))
    recall = round(recall_score(y_test, y_pred), 4)
    # print(('precision_score(y_test, y_pred)))
    precision = round(precision_score(y_test, y_pred), 4)
    # print(f1_score(y_test, y_pred))
    f1_Score = round(f1_score(y_test, y_pred), 4)
    # brier_score
    # predict probabilities
    probs = clf.predict_proba(X_test)
    probs = probs[:, 1]
    loss = round(brier_score_loss(y_test, probs), 4)

    # pandas dataframe to html table flask
    # uploaded_df_html = uploaded_df.to_html()
    # return render_template('show_csv_data.html', data_var=uploaded_df_html)
    return render_template('index2.html', accuracy_score='Accuracy Score = {}'.format(accuracy),
                           recall_score='Recall Score = {}'.format(recall), precision_score='Precision Score = {}'.format(precision), f1_score='F1 Score = {}'.format(f1_Score),
                           barier_score='Barier Score = {}'.format(loss))


if __name__ == '__main__':
    app.run(debug=True)
