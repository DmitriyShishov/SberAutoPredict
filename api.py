import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
with open('model/sber_auto_prod.pkl', "rb") as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: object
    client_id: object
    visit_date: object
    visit_time: object
    visit_number: int
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    utm_keyword: object
    device_category: object
    device_os: object
    device_brand: object
    device_model: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object


class Prediction(BaseModel):
    client_id: object
    Result: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {
        'client_id': form.client_id,
        'Result': y[0]
    }