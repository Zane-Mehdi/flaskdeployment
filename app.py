import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
fish_set = pd.read_csv('Fish.csv')
fish_category = list(dict.fromkeys(fish_set["Species"]))

@app.route('/')
def hello_world():  # put application's code here
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('index.html', prediction_text=' fish is '+ fish_category[prediction[0]])


if __name__ == '__main__':
    app.run()
