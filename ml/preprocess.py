from flask import json
import os
import numpy as np 
import joblib
import pandas as pd

scaler_new = joblib.load(open(os.getcwd()+'/ml/scaler_new', 'rb'))

def transform(data):
    """
    This method will transfrom raw request data into required format. 
    """
    
    data = parse(data)
    pre_processed = {
      'age': data['age'], 
      'sex': data['sex'],
      'resting_blood_pressure': data['trestbps'],
      'cholesterol': data['chol'],
      'fasting_blood_sugar':data['fbs'],
      'maximum_heart_rate': data['thalach'],
      'exercise_induced_angina': data['exang'],
      'st_depression': data['oldpeak'], 
      'cp_asymptomatic': 1 if data['cp'] == 3 else 0,
      'cp_atypical angina': 1 if data['cp'] == 1 else 0,
      'cp_non-anginal pain':1 if data['cp'] == 2 else 0,
      'cp_typical angina': 1 if data['cp'] == 0 else 0,
      'thal_fixed defect': 1 if data['thal'] == 2 else 0,
      'thal_normal':1 if data['thal'] == 1 else 0, 
      'thal_reversable defect': 1 if data['thal'] == 3 else 0, 
      'ST-T wave abnormality': 1 if data['restecg'] == 1 else 0,
      'left ventricular hypertrophy':1 if data['restecg'] == 2 else 0,
      'normal': 1 if data['restecg'] == 0 else 0,
      'slope_downsloping': 1 if data['slope'] == 2 else 0,
      'slope_flat':1 if data['slope'] == 1 else 0, 
      'slope_upsloping': 1 if data['slope'] == 0 else 0,
      'majore_vessels_0':  1 if data['ca'] == 0 else 0,
      'majore_vessels_1': 1 if data['ca'] == 1 else 0,
      'majore_vessels_2': 1 if data['ca'] == 2 else 0, 
      'majore_vessels_3': 1 if data['ca'] == 3 else 0

    }

    return pre_processed

def scaler(raw): 
    return scaler_new.transform(raw)

def dump(data): 
    return json.dumps(data)    

def parse(string): 
    return json.loads(string)



    