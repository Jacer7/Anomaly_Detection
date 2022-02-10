from nbformat import write
import streamlit as st
import sys
import requests
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
import os


sys.path.insert(0, '../api')
#sys.path.insert(0, '../pipelines')

url = 'http://127.0.0.1:8000/IsolationForest'
if_url = 'http://127.0.0.1:8000/IsolationForest'

def save_uploadedfile(uploadedfile):
     with open(os.path.join("data/uploads",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())

def main():
    st.title("Anomaly Detection Web App")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    st.write('Select the Models for Anomaly Detection:')
    isolation_forest = st.checkbox('Isolation Forest (IF)')
    lof = st.checkbox('Local Outlier Factor (LOF)')
    
    if st.button("Check Anomalies"):
        if uploaded_file is not None:
                
                # Can be used wherever a "file-like" object is accepted:
                # dataframe = pd.read_csv(uploaded_file)
                # dataframe.drop(dataframe.columns[[0]], axis = 1, inplace = True)
                # dataframe.drop(dataframe.iloc[:, 2:53], inplace = True, axis = 1)

                # dataframe = dataframe.dropna()
                # dataframe = dataframe.head(30000)
                # df_dict = dataframe.to_dict(orient='records')

                # x = requests.post(url, json=df_dict)
                # data = x.text
                # df = pd.read_json(data, orient='records')

                # dfBroken = df[df['machine_status']=='BROKEN']
                # dfSensors = df.drop(['machine_status'], axis=1)
                # sensorNames=dfSensors.columns
                # for sensor in sensorNames:
                #     sns.set_context('talk')
                #     fig = plt.figure(figsize=(10,3))
                #     _ = plt.plot(dfBroken[sensor], linestyle='none', marker='X', color='red', markersize=12)
                #     _ = plt.plot(df[sensor], color='grey')
                #     _ = plt.title(sensor)
                #     st.pyplot(fig)

                #st.write(df)

                #uploaded file details
                file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
                #save the uploaded file
                save_uploadedfile(uploaded_file)
                org_df  = pd.read_csv(uploaded_file)
                # org_df.drop(org_df.columns[[0]], axis = 1, inplace = True)
                # org_df.drop(org_df.iloc[:, 2:53], inplace = True, axis = 1)

                # org_df = org_df.dropna()
                
                org_df['timestamp'] = pd.to_datetime(org_df['timestamp'])
                org_df = org_df.set_index('timestamp')
                # st.write("Original Dataframe")
                # st.dataframe(org_df)
                st.write("Original Anomalies")
                dfBroken = org_df[org_df['machine_status']=='BROKEN']
                dfSensors = org_df.drop(['machine_status'], axis=1)
                sensorNames=dfSensors.columns
                for sensor in sensorNames:
                    sns.set_context('talk')
                    fig = plt.figure(figsize=(10,3))
                    _ = plt.plot(dfBroken[sensor], linestyle='none', marker='X', color='red', markersize=12)
                    _ = plt.plot(org_df[sensor], color='grey')
                    _ = plt.title(sensor)
                    st.pyplot(fig)

                #sending the uploaded file to fastapi for anomaly detecion
                files = {'sensor_data': open('./data/uploads/'+file_details['FileName'], 'rb')}
                res = requests.post(if_url, files=files)
                data = res.text
                anomalies_df = pd.read_json(data, orient='records')
                #st.success(res.text)
                #st.write("Anomalies Dataframe")
                #st.dataframe(anomalies_df)

                st.write("Detected Anomalies")
                a = anomalies_df.loc[anomalies_df['PredictedAnamoly'] == -1] #anomaly
                fig = plt.figure(figsize=(18,6))
                _ = plt.plot(anomalies_df['sensor_00'], color='grey', label='Normal')
                _ = plt.plot(a['sensor_00'], linestyle='none', marker='X', color='blue', markersize=12, label='Forest Anomaly', alpha=0.3)
                _ = plt.plot(dfBroken['sensor_00'], linestyle='none', marker='X', color='red', markersize=12, label='Broken')
                _ = plt.xlabel('Date and Time')
                _ = plt.ylabel('Sensor Reading')
                _ = plt.title('sensor_09 FOREST ISOLATION Anomalies')
                _ = plt.legend(loc='best')
                st.pyplot(fig)

                # with open('./data/sensor.csv', 'rb') as f:
                #     r = requests.post(upload_url, files={'./data/sensor.csv': f})
                #     st.success(r.text)
                # # with open('./data/sensor.csv', 'rb') as f:
                #     r = requests.post(upload_url, files={'file': ('sensor.csv', f, 'text/csv', {'Expires': '0'})})
                #     #r = requests.post(url, files={'sensor.csv': f})
                #     #print(r.text)
                #     st.success(r.text)

                # files = {'file_upload': uploaded_file.getvalue()}
                # x = requests.post(url, files=files)

                # st.success(x.status_code)


if __name__ == '__main__':
    main()