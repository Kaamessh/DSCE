from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import load_artifacts, predict_supply_demand, models_healthcheck, ArtifactsLoadError
from services.chatbot import get_chat_response

app = FastAPI(title="Food AI Prediction API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    STATE: str = Field(..., description="Encoded STATE value")
    District_Name: str = Field(..., alias="District Name", description="Encoded District value")
    Market_Name: str = Field(..., alias="Market Name", description="Market name")
    Commodity: str
    Variety: str
    Grade: str
    month: int
    year: int

    class Config:
        allow_population_by_field_name = True


class ChatRequest(BaseModel):
    message: str
    history: list | None = None


def _load_code_name_map() -> Dict[str, Dict[str, str]]:
    json_path = Path(__file__).parent / "models" / "code_name_map.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except Exception:
            pass

    txt_path = Path(__file__).parent / "encoded_mappings.txt"
    if not txt_path.exists():
        return {}

    section_to_field = {
        "STATE": "STATE",
        "District Name": "District Name",
        "Commodity": "Commodity",
    }
    current_field = None
    mapping: Dict[str, Dict[str, str]] = {k: {} for k in section_to_field.values()}

    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("--- Encoded Values and Original Names for "):
            title = line.replace("--- Encoded Values and Original Names for ", "").replace(" ---", "").strip()
            current_field = section_to_field.get(title)
            continue
        if current_field and line.startswith("Encoded Value:"):
            try:
                parts = line.split(",")
                code_part = parts[0].split(":", 1)[1].strip()
                name_part = parts[1].split(":", 1)[1].strip()
                mapping[current_field][code_part] = name_part
            except Exception:
                continue

    return {k: v for k, v in mapping.items() if v}


def _encoder_classes(encoders: Dict[str, Any], name: str) -> List[str]:
    enc = encoders.get(name)
    if enc is None:
        return []
    return [str(v) for v in getattr(enc, "classes_", [])]


@app.get("/health")
def health():
    try:
        load_artifacts()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/options")
def options():
    try:
        artifacts = load_artifacts()
    except (ArtifactsLoadError, FileNotFoundError) as e:
        raise HTTPException(status_code=500, detail=f"Artifacts unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error loading artifacts: {e}")

    encoders = artifacts.get("encoders", {}) if isinstance(artifacts, dict) else {}
    code_name_map = _load_code_name_map()

    option_payload = {
        "STATE": _encoder_classes(encoders, "STATE"),
        "District Name": _encoder_classes(encoders, "District Name"),
        "Market Name": _encoder_classes(encoders, "Market Name"),
        "Commodity": _encoder_classes(encoders, "Commodity"),
        "Variety": _encoder_classes(encoders, "Variety"),
        "Grade": _encoder_classes(encoders, "Grade"),
    }
    return {"options": option_payload, "codeNameMap": code_name_map}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        data = {
            "STATE": payload.STATE,
            "District Name": payload.District_Name,
            "Market Name": payload.Market_Name,
            "Commodity": payload.Commodity,
            "Variety": payload.Variety,
            "Grade": payload.Grade,
            "month": payload.month,
            "year": payload.year,
        }
        result = predict_supply_demand(data)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except (ArtifactsLoadError, FileNotFoundError) as e:
        raise HTTPException(status_code=500, detail=f"Artifacts unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/healthcheck")
def healthcheck():
    hc = models_healthcheck()
    return hc


@app.post("/chat")
def chat(payload: ChatRequest):
    try:
        history = payload.history or []
        reply = get_chat_response(payload.message, history)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
