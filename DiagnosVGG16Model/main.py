from instanceModel1 import *
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000/diagnostic",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["get,post"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"API to Diagnos19, Using the model to Ray X diagnostic"}

@app.post("/image/addres_img")
def lad_image(url):
    diagnos = diagnos19(url)
    diagnostico = int(diagnos[0])
    precision = round(float(diagnos[1]), 4)
    return {"diagnostico": diagnostico, "accuracy_score": precision}

# Funcion que permite usar el modelo
def diagnos19(url):
    diagnostic = InstaceModel.VGG(url)
    resp = [diagnostic[0], diagnostic[1]]
    return resp
