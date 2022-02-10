from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import StringIO
import pickle


from pipeline.preprocess import get_preprocessed

isolation_forest = FastAPI()

# Routes
@isolation_forest.get("/")
async def index():
   return {"api_name": "Isolation Forest"}


class sensor_data(BaseModel):
    timestamp : pd.Timestamp
    sensor_values : float
    machine_status : str
    
@isolation_forest.post("/check_anomalies")
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
   
   #set timestamp as index
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   df = df.set_index('timestamp')

   return df

@isolation_forest.post("/IsolationForest/")
async def create_upload_file_if(sensor_data: UploadFile = File(...)) -> pd.DataFrame:
   #read from csv
   dframe = pd.read_csv(StringIO(str(sensor_data.file.read(), 'utf-8')), encoding='utf-8')
   #preprocess data
   dframe = get_preprocessed(dframe)
   #define x
   X = dframe.iloc[:, 0:1]
   #load model
   loaded_model = pickle.load(open('./models/model_if.pkl', 'rb'))
   #make prediction
   y_pred = loaded_model.predict(X)
   #concatenate the anomalies to the df
   res = pd.concat([X.reset_index(), pd.DataFrame(data=y_pred, columns=['PredictedAnamoly'])], axis=1)
   res['timestamp'] = pd.to_datetime(res['timestamp'])
   res = res.set_index('timestamp')

   res['PredictedAnamoly'] = res['PredictedAnamoly'].map(
                   {1:'1' , -1:'-1'})

   res['machine_status'] = dframe['machine_status']
   print(res)

   #save the df with anomalies in a csv
   filepath = "./data/uploads/if.csv"
   res.to_csv(filepath, index=False)
   return {"csv": filepath}