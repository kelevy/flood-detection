"""
FastAPI inference app for the flood detection U-Net.

Exposes a single POST /predict endpoint that accepts a Sentinel-1 SAR
GeoTIFF (2-band: VV, VH) and returns a predicted flood mask as a
single-band GeoTIFF (0: not water, 1: water), preserving the original
georeferencing (CRS, transform) so the output can be loaded directly
into GIS software (QGIS, rasterio, etc.).

Run locally:
    uvicorn app:app --reload

Then test with:
    curl -X POST "http://localhost:8000/predict" \
         -F "file=@sample_chip.tif" \
         --output prediction.tif
"""

import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import rasterio
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from model import build_model


app = FastAPI(
    title="Flood Detection API",
    description="Predicts flood extent from Sentinel-1 SAR imagery using a U-Net trained on Sen1Floods11.",
    version="1.0",
)

MODEL = None
DEVICE = None


@app.on_event("startup")
def load_model():
    global MODEL, DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    checkpoint_path = os.environ.get("MODEL_CHECKPOINT", "../models/best_model.pt")
    MODEL = build_model().to(DEVICE)
    MODEL.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    MODEL.eval()
    print(f"Model loaded on {DEVICE} from {checkpoint_path}")


def preprocess_sar(file_bytes):
    """
    Load a 2-band SAR GeoTIFF from bytes, normalize (matching training
    preprocessing), and return the array plus the source georeferencing
    metadata needed to write a matching output GeoTIFF.
    """
    with rasterio.io.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img = src.read()  # (2, H, W)
            profile = src.profile.copy()

    if img.shape[0] != 2:
        raise ValueError(f"Expected 2-band SAR image (VV, VH), got {img.shape[0]} bands")

    img_norm = np.nan_to_num(img, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
    img_norm = np.clip(img_norm, -50, 1)
    img_norm = (img_norm + 50) / 51.0

    return img_norm.astype(np.float32), profile


def mask_to_geotiff_bytes(mask, profile):
    """Write the predicted mask as a single-band GeoTIFF, reusing the input's CRS/transform."""
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype="uint8",
        nodata=None,
    )

    buf = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**out_profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
        buf.write(memfile.read())
    buf.seek(0)
    return buf


@app.get("/")
def root():
    return {
        "message": "Flood Detection API",
        "usage": "POST a 2-band Sentinel-1 SAR GeoTIFF (VV, VH) to /predict",
        "returns": "A single-band GeoTIFF (0: not water, 1: water) with matching georeferencing",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        file_bytes = await file.read()
        image, profile = preprocess_sar(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process input: {e}")

    input_tensor = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = MODEL(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    tif_buf = mask_to_geotiff_bytes(pred, profile)
    return StreamingResponse(
        tif_buf,
        media_type="image/tiff",
        headers={"Content-Disposition": "attachment; filename=prediction.tif"},
    )