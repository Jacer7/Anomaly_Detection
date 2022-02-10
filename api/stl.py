from fastapi import FastAPI, File, UploadFile

from pipeline.preprocess import get_sensor_data, sample_sensor_data, get_indexed_df
from pipeline.training import stl_model, get_anomaly_limits, get_anomalies


stl = FastAPI()

# Routes
@stl.get("/")
async def index():
   return {"api_name": "STL Decomposition"}


@stl.post("/STL/")
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
