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

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import brier_score_loss

    # by convention our package is to be imported as dp (dp for Differential Privacy!)
    import numpy as np
    import matplotlib.pyplot as plt

    # flask demo ori model........
    # dataset = pd.read_csv('pima diabetes(email).csv')

    # flask demo ori model........
    # dataset = pd.read_csv('hiring.csv')

    X = uploaded_df.drop('Outcome', axis=1)
    y = uploaded_df[['Outcome']]

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)
    # -------------------------------------#

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.33, random_state=42)
    print(X_train.shape)  # type: ignore
    print(X_test.shape)   # type: ignore

    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    r_squared = round(model.score(X_encoded, y), 4)
    print(r_squared)

    adj_r_squared = round(1 - (1-model.score(X_encoded, y)) *
                          (len(y)-1)/(len(y)-X_encoded.shape[1]-1), 4)
    print(adj_r_squared)

    # predict probabilities
    probs = model.predict_proba(X_test)
    # keep the predictions for class 1 only
    probs = probs[:, 1]
    # calculate bier score
    loss = round(brier_score_loss(y_test, probs), 4)
    print(loss)

    # pandas dataframe to html table flask
    # uploaded_df_html = uploaded_df.to_html()
    # return render_template('show_csv_data.html', data_var=uploaded_df_html)
    return render_template('index3.html', r_squared_value='R_Squared Value = {}'.format(r_squared),
                           adj_r_squared_value='Adjust R_Squared Value = {}'.format(adj_r_squared), barier_score='Barier Score = {}'.format(loss))


if __name__ == '__main__':
    app.run(debug=True)
