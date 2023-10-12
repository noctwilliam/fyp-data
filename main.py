from fastapi import FastAPI, Body
from pydantic import BaseModel
from enum import Enum
import joblib
import numpy as np
import pandas as pd
class Symptoms(BaseModel):
	cough: int
	muscle_aches: int
	tiredness: int
	sore_throat: int
	runny_nose: int
	stuffy_nose: int
	fever: int
	nausea: int
	vomiting: int
	diarrhea: int
	shortness_of_breath: int
	difficulty_breathing: int
	loss_of_taste: int
	loss_of_smell: int
	sneezing: int

features = ['cough', 'muscle_aches', 'tiredness', 'sore_throat', 'runny_nose', 'stuffy_nose', 'fever', 'nausea', 'vomiting', 'diarrhea', 'shortness_of_breath', 'difficulty_breathing', 'loss_of_taste', 'loss_of_smell', 'sneezing']
features = [feature.upper() for feature in features]

app = FastAPI()

@app.get("/")
async def root():
	return {"Influenza Severity Classification"}

@app.post("/predict")
async def predict_symptoms(symptoms: Symptoms):
    symptoms = symptoms.model_dump()
    symptoms = list(symptoms.values())
    output = pd.DataFrame([symptoms], columns=features)
    model = joblib.load('decision_tree_model.pkl')
    prediction = model.predict(output)
    
    return {"Severity": prediction[0].item()}