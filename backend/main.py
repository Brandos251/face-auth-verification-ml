from fastapi import FastAPI
import tensorflow as tf
import cv2

app = FastAPI()

@app.get("/")
async def root():
    return {
        "FastAPI": "работает",
        "TensorFlow": tf.__version__,
        "OpenCV": cv2.__version__
    }