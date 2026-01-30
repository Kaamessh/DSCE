# DSCE

Full-stack (FastAPI + React/Vite) app for predicting agriculture supply vs demand with a chatbot helper and production-style validation.

## Features
- FastAPI endpoints: `/options`, `/predict`, `/chat`, `/health`, `/healthcheck` using the models in `models/`
- React UI (Vite) with EN/TA/HI labels, 5-crop charts over 20 years, and a floating chatbot popup
- TinyLlama LoRA advisor ready: adapter weights are not committed; configs live under `llm/`
- Scripts under `scripts/` for artifact validation, downloads, and conversions

## Backend setup
1. `python -m venv .venv && .venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Run: `uvicorn api:app --reload --host 0.0.0.0 --port 8000`

## Frontend setup
1. `cd frontend`
2. `npm install`
3. `npm run dev -- --host --port 5181` (or omit flags for the default Vite port)
4. Set `VITE_API_URL` if the backend is not at `http://localhost:8000`

## Usage
- Prediction UI: open the frontend dev server, fill fields, view charts, and chat
- Health: `curl http://localhost:8000/health`
- Options: `curl http://localhost:8000/options`
- Chat: `curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\":\"Hi\"}"`

## Models and data
- `models/` holds supply, demand, and price artifacts (PKL/JSON) plus encoders
- Large assets are ignored: add your own `llm/tinyllama_agri_adapter/adapter_model.safetensors` and Hugging Face caches locally
- `encoded_mappings.txt` provides code-to-name lookups for categories

## Notes
- Keep virtualenvs, node_modules, and HF caches out of git (see `.gitignore`)
- For production, build the UI with `npm run build` and serve the static output behind your API
