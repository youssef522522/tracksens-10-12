from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO model
MODEL_PATH = r"E:\uni\grad proj\model\model from first visit from partially labeled data\best.pt"
print(f"Loading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")

@app.get("/")
async def root():
    return {"status": "AI Inference Server Running", "model": MODEL_PATH}

@app.get("/ping")
async def ping():
    print("Ping received from browser!")
    return {"message": "pong"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Log that we received a request
        print(f"Processing frame: {file.filename}")

        # Read image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run inference with very low confidence for testing
        # Lowering conf to 0.01 to ensure we see SOMETHING if it's working
        results = model.predict(source=img, conf=0.01, verbose=False)

        detections = []
        for r in results:
            # Log how many boxes were found by the model
            print(f"  - Model found {len(r.boxes)} potential objects")

            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                # Bounding box coordinates
                bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]

                severity = "high" if conf > 0.7 else "medium" if conf > 0.4 else "low"

                detections.append({
                    "type": label,
                    "confidence": int(conf * 100),
                    "severity": severity,
                    "bbox": bbox
                })

        print(f"  - Returning {len(detections)} detections")
        return {"detections": detections}
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://https://sense-personally-abroad-oscar.trycloudflare.com")
    uvicorn.run(app, host="0.0.0.0", port=8000)
