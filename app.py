import os
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Burn Classifier")

# Load the trained model
MODEL_PATH = os.path.join("models", "BurntSkinClassifier.h5")
model = load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ["First-degree burn", "Second-degree burn", "Third-degree burn"]

# Image size for model input
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(img: Image.Image):
    """Preprocess image for model prediction."""
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process image
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        
        # Preprocess
        processed_img = preprocess_image(img)
        
        # Predict
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
    import os
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
