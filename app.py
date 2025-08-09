"""
Web application for brain tumor segmentation.

This application uses FastAPI as the backend framework to expose a
simple web interface allowing users to upload a single MRI slice
image.  The uploaded image is processed by a U‑Net based model
defined in ``model.py``.  If a trained model file is provided in
``weights/weights.pt``, it will be loaded at startup.  If the model
cannot be loaded (for example if PyTorch is unavailable), the
application falls back to a simple thresholding algorithm to
generate a pseudo segmentation mask.  The segmentation mask is
overlaid onto the original image using a red colormap and saved to
the ``static/outputs`` directory.  The resulting page displays both
the uploaded image and the segmentation result side by side.

References:
* Yousef et al. showed that integrating Atrous Spatial Pyramid
  Pooling (ASPP) and squeeze–excitation residual blocks within a
  Bridged U‑Net architecture improves tumor segmentation accuracy【133007765279740†L314-L331】.
* The original model presented in their paper is used here as a
  blueprint for ``model.py``.  In the absence of trained weights the
  thresholding fallback provides a reasonable proxy for segmenting
  bright tumor regions in MRI images.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

try:
    import torch  # type: ignore
    from model import load_model  # type: ignore
except Exception:
    # PyTorch is not available; ``load_model`` will be None
    torch = None  # type: ignore
    load_model = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _load_segmentation_model() -> Optional[object]:
    """Attempt to load a trained PyTorch model if available.

    If a weights file exists under ``weights/weights.pt`` and both
    PyTorch and ``model.load_model`` are importable, then a model is
    loaded.  Otherwise ``None`` is returned.
    """
    weights_file = BASE_DIR / "weights" / "weights.pt"
    if load_model is None or not weights_file.exists():
        return None
    try:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        return load_model(str(weights_file), device=device)
    except Exception:
        return None


# Instantiate model at startup
MODEL = _load_segmentation_model()


def segment_with_model(image: Image.Image) -> np.ndarray:
    """Perform segmentation using the loaded PyTorch model.

    The input image is resized to 256×256, normalized to [0, 1] and
    converted to a tensor of shape (1, 1, H, W).  The model returns a
    single‑channel probability map; pixels greater than 0.5 are
    considered tumor.  If the model is not loaded, a simple
    thresholding fallback is used instead.
    """
    # Convert to grayscale
    gray = image.convert("L")
    # Resize to model input size
    size = (256, 256)
    gray_resized = gray.resize(size, resample=Image.BILINEAR)
    # Convert to numpy array
    arr = np.asarray(gray_resized, dtype=np.float32) / 255.0
    if MODEL is not None and torch is not None:
        # Prepare tensor
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # shape: (1,1,H,W)
        tensor = tensor.to(next(MODEL.parameters()).device)
        with torch.no_grad():
            pred = MODEL(tensor)
        mask = pred.squeeze().cpu().numpy() > 0.5
    else:
        # Fallback: Otsu-like threshold based on mean and std
        threshold = arr.mean() + 0.5 * arr.std()
        mask = arr > threshold
    # Resize mask back to original image size
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
    return np.array(mask_img) > 0


def overlay_mask_on_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Overlay a binary mask on the input image using a red tint.

    The mask should be a boolean array with the same width and height
    as the input image.  Tumor pixels are highlighted in semi‑transparent
    red, while the original image remains visible underneath.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    overlay = image.copy()
    red = Image.new("RGB", image.size, (255, 0, 0))
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    overlay = Image.composite(red, overlay, mask_img)
    blended = Image.blend(image, overlay, alpha=0.4)
    return blended


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, result_image: str | None = None) -> HTMLResponse:
    """Render the home page with an optional segmentation result."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result_image": result_image},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    """Handle image upload and return segmentation result page."""
    # Save uploaded file to disk
    contents = await file.read()
    input_filename = f"{uuid.uuid4().hex}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename
    with open(input_path, "wb") as f:
        f.write(contents)
    # Open the image
    image = Image.open(input_path)
    # Perform segmentation
    mask = segment_with_model(image)
    # Overlay mask on original image
    overlay_img = overlay_mask_on_image(image, mask)
    # Save result
    output_filename = f"{uuid.uuid4().hex}_seg.png"
    output_path = OUTPUT_DIR / output_filename
    overlay_img.save(output_path)
    # Redirect to home page with result
    return RedirectResponse(url=f"/?result_image=outputs/{output_filename}", status_code=303)