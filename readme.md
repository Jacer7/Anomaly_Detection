# Install Libraries
```bash
pandas==1.2.4
numpy==1.20.1
matplotlib==3.3.4
sklearn
scikit-learn==0.24.1
streamlit==1.2.0
fastapi==0.70.0
uvicorn==0.15.0
aiohttp==3.8.1
python-multipart==0.0.5
seaborn==0.11.1
gunicorn==20.1.0
statsmodels==0.13.2

```
# Frontend Libraries
<h3>Streamlit</h3>
Please read the following guidelines for the <b>Streamlit</b> Setup:<br>
https://docs.streamlit.io/library/get-started/installation<br><br>

```angular2html
pip install streamlit
```

# Fast Api Backend
<h3>FastAPI</h3>
Please read the following guidelines for the <b>FastAPI</b> Setup:<br>
https://fastapi.tiangolo.com/tutorial/<br><br>

```angular2html
pip install fastapi
pip install uvicorn
pip install python-multipart
```


# RUN THIS APP LOCALLY
To run this app locally, clone the code from the <b>local branch</b> (very important). Then, set up the virtual environment in your system and run the following command:<br>
```angular2html
pip install -r requirements.txt
```
After that, run the following below servers:

# RUN STREAMLIT SERVER
```angular2html
streamlit run web_app/frontend.py
```

# RUN SERVER FASTAPI
For Isolation Forest:
```angular2html
uvicorn api.if:ifr --reload
```
For LOF:
```angular2html
uvicorn api.lof:lof --reload
```
For STL Decomposition:
```angular2html
uvicorn api.stl:stl_decomposition --reload
```


Now your app should be running on your localhost with the port 8501 depending upon your system (please check the streamlit terminal). You can access it most probably with the following link:
http://localhost:8501/ or http://127.0.0.1:8501/

<br>
<h1>Streamlit Cloud Deployment</h1>
The web app has been deployed on Streamlit Cloud. You can go ahead and check it out on the following link:<br>
https://share.streamlit.io/wakarhassanalvi/time-series-anomaly-detection/web_app/frontend.py
<br><br>
For each of the Isolation Forest, LOF and STL Decomposition - this web app makes requests to <b>Heroku Endpoints</b> of each of the API's to get the anomalies as response.
<br>

<h1>Heroku Deployment</h1>
For each of the models i.e. Isolation Forest, LOF and STL Decomposition - we have deployed 3 API's for each model on Heroku. The Heroku endpoints for each of the models are listed below:<br><br>
1. For Isolation Forest - https://ts-anomaly-detection-if.herokuapp.com/docs<br>
2. For LOF - https://ts-anomaly-detection-lof.herokuapp.com/docs<br>
3. For STL Decomposition - https://ts-anomaly-detection-stl.herokuapp.com/docs<br><br>

The deployed Streamlit Cloud Web App makes the API calls to these Heroku API endpoints to get the anomalies. If you want to check these endpoints individually, just click on the respective endpoint and try out the post method by uploading the test csv file.
