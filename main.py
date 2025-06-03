import os
import numpy as np
from model import OnnxImageClassifier, Preprocessor
import io

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


def load_model():
    """This function initalizes loading the model and preprocessor outside of the predict function
       as a form of optimisation where the predict function doesn't have to always load the model on 
       each call or when the endpoint is hit. loading the model takes abit of time and is done once
       here.
    """
    try:
        model_path = "model.onnx"

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")

        print("Loading ONNX model...")
        classifier = OnnxImageClassifier(model_path)
        print("Done loading model")

        #initializing preprocessor
        preprocessor = Preprocessor()

    except Exception as e:
        print(f"failed to load model: {str(e)}")
        classifier = None
        preprocessor = None

    return classifier, preprocessor


classifier, preprocessor = load_model()

@app.get("/health")
def health():
    return "OK"

@app.post("/predict")
async def predict(model_bytes_str: UploadFile = File(...)):

    model_inputs = await model_bytes_str.read()
    image_tensor = preprocessor.preprocess(io.BytesIO(model_inputs))
    try:
        print("getting prediction....")
        class_id = classifier.predict(image_tensor)

        result = {"class_id": class_id}

        print("obtained prediction")
    except:
        print("classification failed")
        result = {"error": 0}

    return result
    



