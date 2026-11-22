"""Clean FastAPI entry point for Consumer Complaint Classification."""

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from classifier import DEPARTMENTS, ComplaintClassifier, get_benchmark_rows

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="ClearRoute Complaint Intelligence")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

classifier: ComplaintClassifier | None = None
load_error: str | None = None

try:
    classifier = ComplaintClassifier(BASE_DIR / "saved_model")
except Exception as exc:
    load_error = str(exc)


class ComplaintRequest(BaseModel):
    text: str


@app.get("/")
def serve_home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/status")
def api_status():
    return {
        "ready": classifier is not None,
        "model_name": classifier.model_name if classifier else None,
        "error": load_error,
        "departments": DEPARTMENTS,
    }


@app.get("/api/benchmark")
def api_benchmark():
    return {"rows": get_benchmark_rows(BASE_DIR / "saved_model")}


@app.post("/api/predict")
def api_predict(payload: ComplaintRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Please enter a complaint narrative.")
    return classifier.predict(payload.text.strip())


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
