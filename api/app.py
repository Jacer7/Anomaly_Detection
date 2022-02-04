from fastapi import FastAPI, File, UploadFile
from git import typ
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import StringIO
import pickle


from pipeline.preprocess import get_preprocessed, get_sensor_data, sample_sensor_data, get_indexed_df
from pipeline.training import stl_model, get_anomaly_limits, get_anomalies


app = FastAPI()

# Routes
@app.get("/")
async def index():
   return {"message": "Hello World"}


class sensor_data(BaseModel):
    timestamp : pd.Timestamp
    sensor_00 : float
    machine_status : str
    
@app.post("/check_anomalies")
async def check_anomalies(data: List[sensor_data]) -> pd.DataFrame:
   
   first = data[0]
   received = first.dict()
   df = pd.DataFrame([received])
   dict = {}

   for index, row  in enumerate(data):
      row_data = row.dict()
      #print(index)
      if(index!=0):
         df2 = pd.DataFrame([row_data])
         df = df.append(df2, ignore_index=True)
   
   #anomalies = detect_anomalies(df, MODELS_DIR)
   #print(df)
   #print(first)

   #set timestamp as index
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   df = df.set_index('timestamp')

   return df

@app.post("/IsolationForest/")
async def create_upload_file_if(sensor_data: UploadFile = File(...)) -> pd.DataFrame:
   #read from csv
   dframe = pd.read_csv(StringIO(str(sensor_data.file.read(), 'utf-8')), encoding='utf-8')
   #preprocess data
   dframe = get_preprocessed(dframe)
   #define x
   X = dframe.iloc[:, 0:1]
   #X = X.dropna()
   
   #load model
   loaded_model = pickle.load(open('./models/model_if.pkl', 'rb'))
   #make prediction
   y_pred = loaded_model.predict(X)
   
   res = pd.concat([X.reset_index(), pd.DataFrame(data=y_pred, columns=['PredictedAnamoly'])], axis=1)
   res['timestamp'] = pd.to_datetime(res['timestamp'])
   res = res.set_index('timestamp')

   res['PredictedAnamoly'] = res['PredictedAnamoly'].map(
                   {1:'1' , -1:'-1'})
   #print(res['PredictedAnamoly'].value_counts())

   res['machine_status'] = dframe['machine_status']
   print(res)

   filepath = "./data/uploads/if.csv"
   res.to_csv(filepath, index=False)
   return {"csv": filepath}

@app.post("/LOF/")
async def create_upload_file_lof(sensor_data: UploadFile = File(...)) -> pd.DataFrame:
    #read from csv
   dframe = pd.read_csv(StringIO(str(sensor_data.file.read(), 'utf-8')), encoding='utf-8')
   #preprocess data
   dframe = get_preprocessed(dframe)
   #define x
   X = dframe.iloc[:, 0:1]
   #X = X.dropna()
   
   #load model
   loaded_model = pickle.load(open('./models/model_lof.pkl', 'rb'))
   #make prediction
   y_pred = loaded_model.predict(X)

   res = pd.concat([X.reset_index(), pd.DataFrame(data=y_pred, columns=['PredictedAnamoly'])], axis=1)
   res['timestamp'] = pd.to_datetime(res['timestamp'])
   res = res.set_index('timestamp')

   res['PredictedAnamoly'] = res['PredictedAnamoly'].map(
                   {1:'1' , -1:'-1'})
   #print(res['PredictedAnamoly'].value_counts())

   res['machine_status'] = dframe['machine_status']
   print(res)

   filepath = "./data/uploads/lof.csv"
   res.to_csv(filepath, index=False)
   return {"csv": filepath}


@app.post("/STL/")
async def create_upload_file_stl(coef: str, sensor_data: UploadFile = File(...)):
   
   file = sensor_data.filename
   dfsensor, anomalies = stl_decomposition(sensor_data, 3)
   anomalies = anomalies.rename({'0': 'sensor_values'}, axis=1).reset_index()
   dfsensor = dfsensor.reset_index()
   filepathAnomalies = "./data/uploads/stl.csv"
   filepathSampledSensorData = "./data/uploads/sampledSensorStl.csv"
   # to upload files
   anomalies.to_csv(filepathAnomalies, index=False)
   dfsensor.to_csv(filepathSampledSensorData, index=False)

   return {"anomaly_csv": filepathAnomalies, "sensor_csv": filepathSampledSensorData}


def stl_decomposition(file, coeff):
    df = get_sensor_data(file)
    sampled_df = sample_sensor_data(df)
    #print(sampled_df)
    stlData = stl_model(sampled_df)
    l, u = get_anomaly_limits(stlData.resid, coeff)
    anomalies = get_anomalies(stlData.resid, sampled_df, l, u)
    indexedSensor = get_indexed_df(sampled_df, 0)
    return indexedSensor, anomalies
