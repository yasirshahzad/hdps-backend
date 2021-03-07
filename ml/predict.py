import os
import joblib
import matplotlib.pyplot as plt
import shap 
import pandas as pd
import random, string
from pathlib import Path
import io
from storage import drive


model = joblib.load(open(os.getcwd()+'/ml/dt.gz', 'rb'))

def do_predict(preProcessed): 
    return model.predict(preProcessed)


def do_cause(preProcessed): 
    preProcessed = pd.Series(preProcessed)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(preProcessed)
    shap.initjs()

    letters = string.ascii_lowercase
    id = ''.join(random.choice(letters) for i in range(10))  + '.png'

    buffer = io.BytesIO()

    shap.force_plot(explainer.expected_value[1], shap_values[1], preProcessed, show=False,matplotlib=True).savefig(buffer,format = "png",dpi = 150,bbox_inches = 'tight')
    buffer.seek(0)
    
    image_path = drive.upload_blob(id, buffer)

    buffer.close()
    return id



  