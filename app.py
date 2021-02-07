from flask import Flask, request, url_for, redirect, render_template
from pickle import load
import numpy as np
import pandas as pd
np.random.seed(42)
from src import mylib, make_prediction

app = Flask(__name__)

model=load(open('./models/adaboost_batch_train.pkl','rb'))
required_features=mylib.read_csv_to_list('./data/processed/required_features.csv',header=None,squeeze=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    #features=[x for x in request.form.values()]
    #final=np.array(features)
    #input_df=pd.DataFrame([final], columns=required_features)
    X=[x for x in request.form.values()]
    #y_pred=make_prediction.main(model,X)

    y_pred=make_prediction.main(X[0])

    return y_pred

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3245, debug=True)