from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model('models/mobilenet_final.h5')

# Функция для чтения и предобработки изображения из байтовых данных
def read_imagefile(file_bytes):
    """
    Преобразует байты загруженного файла в numpy-массив,
    меняет размер до 224x224 и нормализует пиксели.
    """
    img = Image.open(BytesIO(file_bytes)).convert('RGB')
    img = img.resize((224, 224), resample=Image.BICUBIC)  # размер входа для MobileNetV2 обычно 224x224
    img_array = np.array(img) / 255.0  # нормализация до [0,1]
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает изображение, выполняет препроцессинг, передаёт в модель и возвращает
    результат в формате текста (процент подлинности).
    """
    try:
        # Читаем содержимое файла (байты)
        data = await file.read()
        # Преобразуем в numpy-формат и добавляем batch-измерение
        image = read_imagefile(data)
        img_batch = np.expand_dims(image, axis=0)

        # Выполняем предсказание с помощью модели
        prediction = MODEL.predict(img_batch)[0]  # одномерный массив с вероятностями
        # Если модель бинарная и выдаёт одно число (вероятность "подлинного"), допустим так:
        confidence = float(prediction[0]) if prediction.size > 1 else float(prediction)
        # Конвертируем в проценты
        percent = int(confidence * 100)

        return {"result": f"Изображение подлинное на {percent}%"}  # Ответ в виде JSON
    except Exception as e:
        # Обработка ошибок при загрузке/обработке файла или предсказании
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

