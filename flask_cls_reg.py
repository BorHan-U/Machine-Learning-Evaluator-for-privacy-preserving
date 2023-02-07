from flask import Flask, render_template, request, session
import pandas as pd
import os
from werkzeug.utils import secure_filename


# Define folder to save uploaded files to process further
# UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads') #For windows
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


# (1)----------------------------------------- Classification Model ----------------------------------------------#


@app.route('/show_result_classification')
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


# (2)----------------------------------------- Regression Model ----------------------------------------------#

@app.route('/show_result_regression')
def showResultreg():
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

    X = uploaded_df.drop('Outcome', axis=1)
    y = uploaded_df[['Outcome']]

    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    # As decision tree algorithm model not taking string as an input, here use "OrdinalEncoder" to convert datasets
    encoder = OrdinalEncoder()
    encoder.fit(X)
    X_encoded = encoder.transform(X)
    # ---------------------#

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


# (3)----------------------------------------- Clustering Model ----------------------------------------------#

@app.route('/show_result_clustering')
def showResultclust():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)

    # libraries
    from sklearn.cluster import KMeans

    # dataset = pd.read_csv("Mall_Customers.csv")
    # df = pd.DataFrame(dataset)
    # print(df)

    # Missing values computation
    # null_value = dataset.isnull().sum()

    # print(null_value)

    # Feature sleection for the model
    # Considering only 2 features (Annual income and Spending Score) and no Label available
    X = uploaded_df.iloc[:, [3, 4]].values

    # Building the Model
    # KMeans Algorithm to decide the optimum cluster number , KMeans++ using Elbow Mmethod
    # to figure out K for KMeans, I will use ELBOW Method on KMEANS++ Calculation
    wcss = []

    # we always assume the max number of cluster would be 10
    # you can judge the number of clusters by doing averaging
    # Static code to get max no of clusters

    for i in range(1, 2):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

        # inertia_ is the formula used to segregate the data points into cluster

    # Visualizing the ELBOW method to get the optimal value of K
    # plt.plot(range(1, 11), wcss)
    # plt.title('The Elbow Method')
    # plt.xlabel('no of clusters')
    # plt.ylabel('wcss')
    # plt.show()
    Inirtia = (str(wcss))

    return render_template('index4.html', Inirtia_value='The Inirtia Value is = {}'.format(Inirtia))


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='206.189.146.29')
