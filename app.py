import streamlit as st
import json
from pathlib import Path

from utils import load_artifacts, predict_supply_demand, models_healthcheck, ArtifactsLoadError
from services.chatbot import get_chat_response, is_claude_sonnet_enabled

st.set_page_config(page_title="AI-Driven Food Supply–Demand Predictor", page_icon="🥦", layout="centered")
st.title("AI-Driven Food Supply–Demand Predictor")

# Informational note about models
with st.expander("About models", expanded=False):
    st.write("""
    This app loads pre-trained models and encoders from the local models/ folder.\
    Current schema (from feature_columns.pkl): STATE, District Name, Market Name, Commodity, Variety, Grade,\
    Min_Price, Max_Price, Modal_Price, Demand _Index, month, year, week, day, arrival_lag_7, arrival_lag_30,\
    arrival_roll_7, arrival_roll_30.
    """)

# Load encoder classes to build select options
try:
    artifacts_preview = load_artifacts()
    encoders = artifacts_preview.get("encoders", {}) if isinstance(artifacts_preview, dict) else {}
except Exception:
    encoders = {}

def _load_code_name_map():
    # Prefer explicit JSON mapping if provided
    json_path = Path(__file__).parent / "models" / "code_name_map.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except Exception:
            pass

    # Fallback: parse encoded_mappings.txt in project root
    txt_path = Path(__file__).parent / "encoded_mappings.txt"
    if not txt_path.exists():
        return {}

    section_to_field = {
        "STATE": "STATE",
        "District Name": "District Name",
        "Commodity": "Commodity",
    }
    current_field = None
    mapping = {k: {} for k in section_to_field.values()}

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

    # Drop empty sections
    mapping = {k: v for k, v in mapping.items() if v}
    return mapping

CODE_NAME_MAP = _load_code_name_map()

def _classes(name: str):
    enc = encoders.get(name)
    if enc is None:
        return []
    return [str(v) for v in getattr(enc, "classes_", [])]

def _fmt(field: str):
    def inner(v):
        if v is None:
            return ""
        name_map = CODE_NAME_MAP.get(field, {})
        label = name_map.get(str(v), str(v))
        if label == str(v):
            return str(v)
        return f"{label} (code {v})"
    return inner

# Prediction form reduced to required fields
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        state = st.selectbox("STATE", options=_classes("STATE"), format_func=_fmt("STATE"))
        district = st.selectbox("District Name", options=_classes("District Name"), format_func=_fmt("District Name"))
        market = st.selectbox("Market Name", options=_classes("Market Name"), format_func=_fmt("Market Name"))
        commodity = st.selectbox("Commodity", options=_classes("Commodity"), format_func=_fmt("Commodity"))
    with col2:
        variety = st.selectbox("Variety", options=_classes("Variety"), format_func=_fmt("Variety"))
        grade = st.selectbox("Grade", options=_classes("Grade"), format_func=_fmt("Grade"))
        month = st.slider("Month", 1, 12, 1)
        year = st.number_input("Year", min_value=1, value=2024, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Ensure artifacts are loadable first
        _ = load_artifacts()
    except (ArtifactsLoadError, FileNotFoundError) as e:
        st.error(f"Model artifacts are not available: {e}")
        hc = models_healthcheck()
        if not hc["ok"]:
            st.warning("Healthcheck findings:")
            for err in hc["errors"]:
                st.write(f"- {err}")
    except Exception as e:
        # Unexpected error surface
        st.error(f"Model artifact loading encountered an unexpected error: {e}")
    else:
        input_data = {
            "STATE": state,
            "District Name": district,
            "Market Name": market,
            "Commodity": commodity,
            "Variety": variety,
            "Grade": grade,
            "month": month,
            "year": year,
        }
        try:
            result = predict_supply_demand(input_data)
            # Metrics/cards
            cards = st.columns(3)
            cards[0].metric("Predicted Price (per 10 kg)", f"{result['price']:.2f}" if result.get("price") is not None else "N/A")
            cards[1].metric("Predicted Supply (kg)", round(result["supply"], 2))
            cards[2].metric("Predicted Demand (kg)", round(result["demand"], 2))

            st.markdown("---")
            st.subheader("Market Insight")
            status_text = result.get("market_status", "")
            decision_text = result.get("decision", "")
            explanation = result.get("explanation", "")

            status_col, decision_col = st.columns(2)
            status_col.info(f"Market Status: {status_text}")
            decision_col.success(decision_text if decision_text else "Decision unavailable")

            st.write(explanation)
            if result.get("using_fallback"):
                st.caption("Supply prediction uses a heuristic fallback due to missing/invalid model.")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.divider()

# Chatbot disabled to keep UI focused on core predictions.

# Chatbot intentionally disabled per requirements to keep UI focused on core predictions.
