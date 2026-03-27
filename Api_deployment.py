from fastapi import FastAPI
import joblib
from pydantic import BaseModel,Field
import pandas as pd
import uvicorn

#Load asset:
model=joblib.load('saved_model/rf_model.pkl')
scaler=joblib.load('saved_model/scaler.pkl')
features=joblib.load('saved_model/features_names.pkl')

#create a structure for getting data.
class Sensor_input_data(BaseModel): 
    #Index(['sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17',
    #    'sensor_2', 'sensor_20', 'sensor_21', 'sensor_3', 'sensor_4',
    #    'sensor_7', 'sensor_8'])
    data:list=Field(
        description='float values of: sensor_11,sensor_12,sensor_13,sensor_15,,sensor_17,sensor_2,sensor_20,,sensor_21,sensor_3,sensor_4,sensor_8',
        min_length=12,
        max_length=12,
        examples=[11.2,22.23,34.4556,64.77888,36.98,45.77,67.88,88.78,12.34,19.88,89.00,56.89])

# from demo import get_train_test_values
# print(get_train_test_values(60,100))

app=FastAPI(title='Engine RUL predictor')
@app.get('/predict')
def predict_rul(input:Sensor_input_data):
    #1.load the data from user:
    df=pd.DataFrame([input.data],columns=features)

    #2.scale the data using scaler:
    scaled_data=scaler.transform(df)
    
    #3.predict:
    prediction=model.predict(scaled_data)
    if prediction[0]<=30:
        return{'Predicted RUL:':prediction[0],'Engine Status:':'Critical Stage'}
    elif prediction[0]>30 and prediction[0]<=100:
        return {'Predicted RUL':prediction[0],'Engine Status':'Maintainance Required'} #model.predict give numpy array so, we used indexing to get the predicted output.
    else:
        return{'Predicted RUL:':prediction[0],'Engine Status:':'Good'}

        
if __name__=='__main__':
    uvicorn.run(app)

