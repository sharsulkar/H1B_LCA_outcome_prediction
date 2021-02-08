from flask import Flask, request, url_for, redirect, render_template
from pickle import load
import numpy as np
import pandas as pd
np.random.seed(42)
from src import mylib, make_prediction

app = Flask(__name__) #name of the flask application

#load the trained ML model from repository
model=load(open('./models/adaboost_batch_train.pkl','rb'))
#load the required features list
required_features=mylib.read_csv_to_list('./data/processed/required_features.csv',header=None,squeeze=True)

@app.route('/')
def home(): #home page - static webpage
    return render_template('home.html')

@app.route('/FAQ')
def fair_warning(): #static webpage showing info on the limitations of the tool and terms of use
    return render_template('terms_of_use.html')

@app.route('/sample_prediction')
def sample_prediction(): #static webpage explaining how to use the tool
    return render_template('sample_prediction.html')

@app.route('/user_input',methods=['POST']) 
def take_user_input():
    return render_template('user_input_form.html')

@app.route('/predict',methods=['POST'])     
def predict():
    #features=[x for x in request.form.values()]
    #final=np.array(features)
    #input_df=pd.DataFrame([final], columns=required_features)
    X=[x for x in request.form.values()]
    #y_pred=make_prediction.main(model,X)

    y_pred=make_prediction.main(X[0])

    return y_pred #render_template('predict_and_explain.html',pred=y_pred)

@app.route('/how_does_it_work')
def how_does_it_work(): #static webpage explaining how prediction is made
    return render_template('how_does_it_work.html')

@app.route('/technical_documentation')
def documentation(): #static webpage that displays autodoc documentation
    return render_template('technical_docs.html')

@app.route('/contact')
def contact(): #static webpage that shows contact information
    return render_template('contact.html')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=3245, debug=True)