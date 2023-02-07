# flask demo ori model........
dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

# Converting words to integer values


def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0}
    return word_dict[word]


X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))

y = dataset.iloc[:, -1]

# Splitting Training and Test Set
# Since we have a very small dataset, we will train our model with all availabe data.

regressor = LinearRegression()

# Fitting model with trainig data
regressor.fit(X, y)


# Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2, 9, 6]]))



@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Accuracy {}'.format(output))


    @app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


    #full html index

    <!DOCTYPE html>
<html>
<!--From https://codepen.io/frytyler/pen/EGdtg-->

<head>
    <meta charset="UTF-8">
    <title>ML Evaluator</title>
    <link href='https://fonts.googleapis.com/css?family=Pacifico' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Arimo' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Hind:300' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Open+Sans+Condensed:300' rel='stylesheet' type='text/css'>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">

</head>

<body>
    <div class="login">
        <h2>Machine Learning Evaluator for Privacy-preserving</h2>

        <!-- Main Input For Receiving Query to our ML -->
        <form action="{{ url_for('predict')}}" method="post">
            <button type="upload Dataset" class="btn btn-primary btn-block btn-large">Upload Dataset</button>
            <!--<input type="text" name="experience" placeholder="Experience" required="required" />
            <input type="text" name="test_score" placeholder="Test Score" required="required" />
            <input type="text" name="experience" placeholder="Experience" required="required" />
            <input type="text" name="test_score" placeholder="Test Score" required="required" />
            <input type="text" name="experience" placeholder="Experience" required="required" />
            <input type="text" name="test_score" placeholder="Test Score" required="required" />
            <input type="text" name="experience" placeholder="Experience" required="required" />
            <input type="text" name="interview_score" placeholder="Interview Score" required="required" />-->


        </form>

        <br>
        <br> {{ prediction_text }}

    </div>


</body>

</html>
