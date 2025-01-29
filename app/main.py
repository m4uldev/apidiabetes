from fastapi import FastAPI, HTTPException
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

load_model = tf.keras.models.load_model('model_prediksi_diabetes.h5')
app = FastAPI()

@app.get
def index():
    return {'Status' : 'Berhasil !!'}

@app.post('/prediksi')
def prediksi(data : dict):
    try:
        pregnancies = data.get('pregnancies')
        glucose = data.get('glucose')
        bloodpressure = data.get('bloodpressure')
        skinthickness = data.get('skinthickness')
        insulin = data.get('insulin')
        bmi = data.get('bmi')
        diabetespedigreefunction = data.get('diabetespedigreefunction')
        age = data.get('age')
        
        scaler = StandardScaler()
        data_deteksi_diabetes = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]])
        data_deteksi_diabetes_scaler = scaler.transform(data_deteksi_diabetes)

        prediksi = load_model.predict(data_deteksi_diabetes_scaler)

        return {"Hasil" : "Berpotensi diabetes." if prediksi > 0.5 else "Tidak berpotensi diabetes."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
