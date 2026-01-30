# DSCE


# 🌾 AI-Driven Food Supply–Demand Predictor  
### Reducing Market Waste • Stabilizing Prices • Empowering Farmers

---

##  Project Overview

Agricultural markets often suffer from **supply–demand imbalance**, leading to:

- -> Overproduction  
- -> Price crashes or spikes  
- -> Post-harvest losses  
- -> Farmer income instability  

This project presents an **AI-powered decision-support system** that predicts **future supply, demand, and price trends** using historical market data and contextual factors — and then **advises whether planting a crop is recommended or not**.

🎯 **Key idea:**  
> Farmers should not enter prices or quantities they don’t know — the system predicts them.

---

##  What Makes This Project Different?

-> Not just prediction — **decision intelligence**  
-> Simple inputs, powerful outputs  
-> Built for **real farmers & market planners**  
-> Explainable and extensible (chatbot-ready)

---

## ** Problem Statement

> Agricultural markets frequently experience mismatches between food supply and consumer demand, resulting in food wastage, price volatility, and income loss for farmers.

**Goal:**  
Build an AI system that:
- Predicts **future supply, demand, and prices**
- Detects **market surplus or shortage**
- Recommends **whether to plant a crop or not**
- Helps reduce **food waste and risk**

---

## 🏗 System Architecture

The system follows a **modular, production-style architecture** where machine learning models, business logic, and user interface are cleanly separated.

The user interacts only with a **React-based frontend**, while all predictions and decisions are handled by an **AI backend**.

---

### 🔹 High-Level Architecture

---

### 🔹 Data Flow Explanation

1. **User Input**  
   The user provides contextual information such as location, crop, season, and weather via the React UI.

2. **API Communication**  
   React sends the input data to the backend using REST APIs in JSON format.

3. **Feature Engineering**  
   The backend converts raw inputs into model-ready features using pre-trained encoders.

4. **ML Predictions**  
   Separate machine learning models predict:
   - Expected Supply
   - Expected Demand
   - Expected Market Price

5. **Decision Logic**  
   A rule-based decision engine determines:
   - Market status (Surplus / Balanced / Shortage)
   - Whether planting the crop is recommended

6. **Response Generation**  
   Results are sent back to the frontend and displayed in a farmer-friendly format.

---

### 🔹 Technology Stack Mapping

7.ML Model Used
  XGBOOST


