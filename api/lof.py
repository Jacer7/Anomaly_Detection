from fastapi import FastAPI, File, UploadFile
import pandas as pd
from io import StringIO
import pickle


from pipeline.preprocess import get_preprocessed

lof = FastAPI()

# Routes
@lof.get("/")
async def index():
   return {"api_name": "LOF"}

@lof.post("/LOF/")
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