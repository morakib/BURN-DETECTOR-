from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from io import BytesIO

app = FastAPI(title="Burn Classifier")

# Load model
MODEL_PATH = os.path.join("models", "BurntSkinClassifier.h5")
model = load_model(MODEL_PATH)

CLASS_NAMES = ["First-degree burn", "Second-degree burn", "Third-degree burn"]
IMG_HEIGHT, IMG_WIDTH = 224, 224

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(img: Image.Image):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Serve HTML
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join("static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

# Predict endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        return JSONResponse({
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

