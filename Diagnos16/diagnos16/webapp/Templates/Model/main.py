from django.http import HttpResponse, JsonResponse

from instanceModel1 import *
from fastapi import FastAPI, File, UploadFile
from typing import Optional
from pydantic import BaseModel
app = FastAPI()

class Item(BaseModel):
    name:str
    price:float
    is_offert: Optional[bool] = None

@app.get("/")
def read_root():
    return {"Hello Word"}

@app.get("/image/{addres_img}")
def lad_image(route):
    diagnos = diagnos19(route)
    return {"Valor": diagnos}
@app.put("/post/{item_id}")
def update_item(item_id: int, item: Item):
    return {"Diagnostico": item.name, "item_id": item_id}

@app.post("/image")
async def image(image: UploadFile = File(...)):
    return {"filename": image.filename}
ruta = r'C:\Users\jr-98\Documents\DataCovid19imgX\TAWSIFUR_RAHMAN\COVID-19_Radiography_DatasetC\TEST\r1.png'
# Funcion que permite usar el modelo
def diagnos19(ruta):
    diagnostic = InstaceModel.VGG(ruta)
    return diagnostic
