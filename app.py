from flask import Flask, render_template, request
import os, joblib, pickle
import numpy as np

app = Flask(__name__)

@app.route("/")

def index():
    return render_template("home.html")

@app.route("/")

def link1():
    return render_template("https://www.verywellhealth.com/stroke-causes-4014093")

@app.route("/result", methods = ['POST','GET'])

def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x = np.array([gender,age,hypertension,heart_disease,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1,-1)
    #scaler_path = os.path.join(r'D:\stroke prediction\Stroke-Prediction-Application', 'scalar.pkl')
    #C:/Users/Riddhi/flaskwork/Stroke-Prediction-Model/
    scaler = pickle.load(open('scalar.pkl', 'rb'))
    '''
    scaler = None

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    '''

    x = scaler.transform(x)
    
    #model_path = os.path.join(r'D:\stroke prediction\Stroke-Prediction-Application', 'finalized_model.pkl')

    #lr = joblib.load(model_path)
    
    lr = pickle.load(open('finalized_model.pkl', 'rb'))
    y_pred_lr = lr.predict(x)
    
    # for no stroke risk
    if y_pred_lr == 0:
        return render_template("nostroke.html")
    else:
        return render_template("stroke.html")

 
if __name__=="__main__":
    app.run(debug=True)
