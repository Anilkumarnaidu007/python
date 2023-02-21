from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sbs
from sklearn import feature_extraction
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
model = pickle.load(open('./models/model.pkl', 'rb'))

@app.route('/')
def index():
   return render_template('index.html')	

@app.route('/predict-single', methods = ['POST'])
def predict_single():
   names_gen=pd.read_csv(r"autogens.csv")
   a = names_gen.sample(frac = 1,random_state=42)
   b = a.dropna(how='all').dropna(how='all',axis=1)
   df2 = b.drop(['Sno', 'Middle Name', 'Language_Main'], axis=1)

   # Get JSON data from request
   json_ = request.json
   # Check if fname is available in json or not
   if 'PAX_FN' not in json_["items"]["features"]:
      return { "Errors": ["Error: First Name not found in JSON"] }, 400
   if 'PAX_LN' not in json_["items"]["features"]:
      return { "Errors": ["Error: Surname not found in JSON"] }, 400

   # Fetch fname and sname
   fname = json_["items"]["features"]['PAX_FN']
   sname = json_["items"]["features"]['PAX_LN']

   df2.loc[len(df2)] = [fname, sname]

   # Get dummies for the categorical variables
   df = pd.get_dummies(df2, columns=['First_Name', 'Surname'], drop_first=True)

   # Get the new columns
   new_cols = [col for col in df.columns if col not in ['First_Name', 'Surname']]

   while len(new_cols) < 4053:
      new_cols.append('dummy')
      df['dummy'] = 0

   # Get the predictions
   predictions = model.predict(df[new_cols])

   # Append prediction in request and send back
   json_["items"]["features"]['language'] = predictions[len(predictions)-1]

   return json_, 200

@app.route('/predict-multi', methods = ['POST'])
def predict_multi():
   names_gen=pd.read_csv(r"autogens.csv")
   a = names_gen.sample(frac = 1,random_state=42)
   b = a.dropna(how='all').dropna(how='all',axis=1)
   df2 = b.drop(['Sno', 'Middle Name', 'Language_Main'], axis=1)

   # Get JSON data from request
   json_ = request.json

   errorOccured = False
   
   for i in range(len(json_["items"])):
      # Check if fname is available in json or not
      if 'PAX_FN' not in json_["items"][i]["features"]:
         json_["items"][i]["features"]["error"] = "Error: First Name not found in JSON"
         json_["items"][i]["features"]["error"] = None
         errorOccured = True
      if 'PAX_LN' not in json_["items"][i]["features"]:
         json_["items"][i]["features"]["error"] = "Error: Surname not found in JSON"
         json_["items"][i]["features"]["error"] = None
         errorOccured = True

      # Fetch fname and sname
      fname = json_["items"][i]["features"]['PAX_FN']
      sname = json_["items"][i]["features"]['PAX_LN']

      df2.loc[len(df2)] = [fname, sname]
      
      # Get dummies for the categorical variables
      df = pd.get_dummies(df2, columns=['First_Name', 'Surname'], drop_first=True)

      # Get the new columns
      new_cols = [col for col in df.columns if col not in ['First_Name', 'Surname']]   
      
      while len(new_cols) < 4053:
         new_cols.append('dummy')
         df['dummy'] = 0

      # Get the predictions
      predictions = model.predict(df[new_cols])

      # Append prediction in request and send back
      json_["items"][i]["features"]['language'] = predictions[len(predictions)-1]

   if errorOccured:
      return json_, 400
   return json_, 200

if __name__ == '__main__':
   app.run(debug = True)