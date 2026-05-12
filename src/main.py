from fastapi import FastAPI

from .Service.Detection_cancer import router as detection_router

app = FastAPI(
    title="Brain Tumor Detection API",
    description="Upload an image to classify brain tumor type and get a Grad-CAM output image.",
    version="1.0.0",
)

app.include_router(detection_router)

@app.get("/")
async def root():
    return {"message": "Brain Tumor Detection API is running"}
