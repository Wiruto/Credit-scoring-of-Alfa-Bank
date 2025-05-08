from flask import Flask, render_template, redirect, request, url_for
import os
import pandas as pd
from blending import blendingClassifier

app = Flask(__name__)
@app.route('/', methods = ["GET"])
def home():
   return render_template('page.html', title='Home')

@app.route('/', methods = ['POST'])
def submit():
   file = request.files['datafile']
   file_path = 'static/data/predict_data.csv'
   file.save(file_path)
   return redirect(url_for('predict'))

@app.route('/predict')
def predict():
   file_path = 'static/data/predict_data.csv'
   if os.path.isfile(file_path):
      data = pd.read_csv(file_path)
      os.remove(file_path)
      prediction = blendingClassifier(data)
      prediction.to_csv('static/prediction/prediction.csv',index=False)
      return render_template('predict.html',data = prediction)
   else:
      return redirect(url_for('home'))

if __name__ == '__main__':
  app.run(host='localhost',port=8000)
   # app.run(host='0.0.0.0', port=8000)