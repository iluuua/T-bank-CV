import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from .schemas import DetectionResponse, Detection, BoundingBox, ErrorResponse
from ..infer.predictor import Predictor
from ..infer.refiner import filter_not_tinkoff

app = FastAPI(title="tbank-logo-detector", version="1.0")
predictor = Predictor.load_from_cfg("configs/infer/default.yaml")

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        im = Image.open(io.BytesIO(content)).convert("RGB") 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    boxes = predictor.predict(im)
    boxes = filter_not_tinkoff(boxes, im)

    detections = [Detection(bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2))
                  for x1, y1, x2, y2 in boxes]
    return DetectionResponse(detections=detections)
