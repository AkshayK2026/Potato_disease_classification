from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from backend.model_loader import load_model, CLASS_NAMES
from backend.utils import preprocess_image
from backend.inference import predict

app = FastAPI(title="Potato Disease Classification API")

model = load_model()

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = preprocess_image(image)

    label, confidence = predict(model, image_tensor, CLASS_NAMES)

    return {
        "disease": label,
        "confidence": round(confidence, 3)
    }
