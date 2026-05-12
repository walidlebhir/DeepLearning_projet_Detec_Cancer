import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

router = APIRouter(prefix="/detection", tags=["detection"])

CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_PATH = Path(__file__).resolve().parent.parent / "System_Deep_learning" / "Model" / "brain_tumor_final_v2.pth"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "static" / "gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(model_path: Path) -> nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Le modèle n'a pas été trouvé: {model_path}")

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_NAMES)),
    )

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except RuntimeError as exc:
        if "Weights only load failed" in str(exc) or "Unsupported global" in str(exc):
            with torch.serialization.safe_globals([np._core.multiarray.scalar]):
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        else:
            raise

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


MODEL = build_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image).unsqueeze(0).to(DEVICE)


def predict(image: Image.Image) -> Dict:
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_index = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])
    probabilities = {CLASS_NAMES[i]: float(p * 100.0) for i, p in enumerate(probs)}

    return {
        "pred_index": pred_index,
        "pred_class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "input_tensor": input_tensor,
    }


def create_gradcam_image(image: Image.Image, input_tensor: torch.Tensor, target_index: int) -> Image.Image:
    image_resized = image.resize((224, 224))
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    target_layer = MODEL.features[-1]
    cam = GradCAM(model=MODEL, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_index)])[0]

    visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
    return Image.fromarray(visualization)


@router.get("/")
async def health_check():
    return {"message": "Detection service is ready"}


@router.post("/predict")
async def predict_cancer(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier uploadé doit être une image.")

    contents = await image.read()
    try:
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Impossible de lire l'image: {exc}")

    prediction = predict(pil_image)
    gradcam_image = create_gradcam_image(pil_image, prediction["input_tensor"], prediction["pred_index"])

    file_name = f"gradcam_{uuid4().hex}.png"
    output_path = OUTPUT_DIR / file_name
    gradcam_image.save(output_path, format="PNG")

    return {
        "filename": image.filename,
        "predicted_class": prediction["pred_class"],
        "confidence": round(prediction["confidence"] * 100.0, 2),
        "probabilities": {k: round(v, 2) for k, v in prediction["probabilities"].items()},
        "gradcam_image_url": f"/detection/gradcam/{file_name}",
    }


@router.post("/predict/png")
async def predict_cancer_png(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier uploadé doit être une image.")

    contents = await image.read()
    try:
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Impossible de lire l'image: {exc}")

    prediction = predict(pil_image)
    gradcam_image = create_gradcam_image(pil_image, prediction["input_tensor"], prediction["pred_index"])

    buffer = BytesIO()
    gradcam_image.save(buffer, format="PNG")
    buffer.seek(0)

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=gradcam_{image.filename}.png"},
    )


@router.get("/gradcam/{file_name}")
async def get_gradcam_image(file_name: str):
    file_path = OUTPUT_DIR / file_name
    if not file_path.exists() or not file_name.endswith(".png"):
        raise HTTPException(status_code=404, detail="Image Grad-CAM introuvable.")
    return FileResponse(path=file_path, media_type="image/png", filename=file_name)


@router.post("/predict/html", response_class=HTMLResponse)
async def predict_cancer_html(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier uploadé doit être une image.")

    contents = await image.read()
    try:
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Impossible de lire l'image: {exc}")

    prediction = predict(pil_image)
    gradcam_image = create_gradcam_image(pil_image, prediction["input_tensor"], prediction["pred_index"])

    buffer = BytesIO()
    gradcam_image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{encoded_image}"

    html_content = f"""
    <html>
      <body>
        <h1>Prédiction : {prediction['pred_class']} ({round(prediction['confidence']*100.0, 2)}%)</h1>
        <img src=\"{data_uri}\" alt=\"GradCAM\" style=\"max-width:100%;height:auto;\" />
        <p>Télécharger l'image : <a href=\"{data_uri}\" download=\"gradcam_{image.filename}.png\">Télécharger PNG</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)
