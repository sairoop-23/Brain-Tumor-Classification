from fastapi import FastAPI, UploadFile, File
import uvicorn
from tensorflow.keras.models import load_model
import numpy as np
from utils import preprocess_image
from io import BytesIO

# Load model
model_path = r"model/resnet50_brain_tumor_model.h5" #add model path from local system
model = load_model(model_path)


class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]

app = FastAPI(title="Brain Tumor API with ResNet50")

@app.get("/")
def home():
    return {"message": "Welcome to Brain Tumor Classification API with ResNet50"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = preprocess_image(contents)
    preds = model.predict(img_array)
    class_names = ["glioma", "meningioma", "no_tumor", "pituitary"]
    result = class_names[np.argmax(preds)]
    return {"prediction": result}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=800, reload=True)

