from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#import pickle models here
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))
application = Flask(__name__)
app=application
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template("home.html")
    else:   
        try:
            #get the data from JSON request
            data = request.get_json()
            
            #extract features in correct order (9 features)
            features = [
                data.get('temperature', 0),
                data.get('rh', 0),
                data.get('ws', 0),
                data.get('rain', 0),
                data.get('ffmc', 0),
                data.get('dmc', 0),
                data.get('isi', 0),
                data.get('classes', 0),
                data.get('region', 0)
            ]
            
            final_features = [np.array(features)]
            
            #scale the features
            scaled_features = standard_scaler.transform(final_features)
            
            #make prediction
            prediction = ridge_model.predict(scaled_features)
            output = round(prediction[0], 2)
            
            #return JSON response
            return jsonify({'prediction': output, 'success': True})
         
        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 400
if __name__=='__main__':
    app.run(host="0.0.0.0")